from __future__ import annotations

import random
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from string_embed.model import (
    DEFAULT_CANDIDATE_COUNT,
    MODEL_PARAMETER_LIMIT,
    NAME_CONTEXT_CHARS,
    StringEmbedder,
    count_parameters,
    encode_text,
    nearest_by_embedding,
    normalize_name,
)


def edit_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, cb in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (ca != cb))
    return row[-1]


def normalized_edit_distance(a: str, b: str) -> float:
    return edit_distance(a, b) / max(len(a), len(b), 1)


def normalized_levenshtein_similarity(a: str, b: str) -> float:
    return 1 - normalized_edit_distance(a, b)


def _token_dice(a: list[str], b: list[str]) -> float:
    if not a and not b:
        return 1.0
    a_set = set(a)
    b_set = set(b)
    return 2 * len(a_set & b_set) / max(len(a_set) + len(b_set), 1)


def _initials(tokens: list[str]) -> str:
    return "".join(token[0] for token in tokens if token)


def name_similarity(a: str, b: str) -> float:
    a_norm = normalize_name(a)
    b_norm = normalize_name(b)
    a_tokens = a_norm.split()
    b_tokens = b_norm.split()
    compact = normalized_levenshtein_similarity(a_norm.replace(" ", ""), b_norm.replace(" ", ""))
    sorted_compact = normalized_levenshtein_similarity("".join(sorted(a_tokens)), "".join(sorted(b_tokens)))
    token = _token_dice(a_tokens, b_tokens)
    initials = normalized_levenshtein_similarity(_initials(a_tokens), _initials(b_tokens))
    return 0.45 * compact + 0.25 * sorted_compact + 0.20 * token + 0.10 * initials


SimilarityFn = Callable[[str, str], float]
T = TypeVar("T")


def _progress(iterable: Iterable[T], enabled: bool, **kwargs) -> Iterable[T] | Iterator[T]:
    if not enabled:
        return iterable

    from tqdm.auto import tqdm

    return tqdm(iterable, **kwargs)


def similarity_matrix(
    words: list[str],
    similarity_fn: SimilarityFn = name_similarity,
    progress: bool = False,
) -> np.ndarray:
    similarities = np.eye(len(words), dtype=np.float32)
    if progress:
        from tqdm.auto import tqdm

        total_pairs = len(words) * (len(words) - 1) // 2
        with tqdm(total=total_pairs, desc="scoring name pairs", unit="pair") as progress_bar:
            for i, a in enumerate(words):
                for j in range(i + 1, len(words)):
                    similarities[i, j] = similarities[j, i] = similarity_fn(a, words[j])
                progress_bar.update(len(words) - i - 1)
        return similarities

    for i, a in enumerate(words):
        for j in range(i + 1, len(words)):
            similarities[i, j] = similarities[j, i] = similarity_fn(a, words[j])
    return similarities


class TripletWords(Dataset):
    def __init__(
        self,
        words: list[str],
        max_len: int = NAME_CONTEXT_CHARS,
        k: int = 8,
        sample_size: int = 64,
        similarity_fn: SimilarityFn = name_similarity,
        progress: bool = False,
    ):
        names = [normalized[:max_len] for word in words if (normalized := normalize_name(word))]
        if len(names) < 3:
            raise ValueError("Need at least three names to train triplets.")
        self.words = names
        self.max_len = max_len
        self.similarity_fn = similarity_fn
        self.k = min(k, max(1, len(self.words) - 1))
        self.sample_size = min(max(sample_size, self.k * 2, 2), len(self.words) - 1)

    def __len__(self) -> int:
        return len(self.words)

    def __getitem__(self, idx: int):
        anchor = self.words[idx]
        candidates = self._sample_candidate_indices(idx)
        scored = sorted(
            ((other, self.similarity_fn(anchor, self.words[other])) for other in candidates),
            key=lambda item: item[1],
            reverse=True,
        )
        pos, pos_sim = random.choice(scored[: self.k])
        neg, neg_sim = random.choice(scored[-self.k :])
        pos_neg_sim = self.similarity_fn(self.words[pos], self.words[neg])
        return (
            encode_text(anchor, self.max_len),
            encode_text(self.words[pos], self.max_len),
            encode_text(self.words[neg], self.max_len),
            torch.tensor(pos_sim, dtype=torch.float32),
            torch.tensor(neg_sim, dtype=torch.float32),
            torch.tensor(pos_neg_sim, dtype=torch.float32),
        )

    def _sample_candidate_indices(self, idx: int) -> list[int]:
        if self.sample_size == len(self.words) - 1:
            return [other for other in range(len(self.words)) if other != idx]

        candidates: set[int] = set()
        while len(candidates) < self.sample_size:
            other = random.randrange(len(self.words))
            if other != idx:
                candidates.add(other)
        return list(candidates)


def triplet_loss(outputs, targets):
    anchor, positive, negative = outputs
    pos_sim, neg_sim, pos_neg_sim = targets
    pos_embed = F.cosine_similarity(anchor, positive)
    neg_embed = F.cosine_similarity(anchor, negative)
    pos_neg_embed = F.cosine_similarity(positive, negative)
    rank = F.relu(neg_embed - pos_embed + (pos_sim - neg_sim))
    mse = (pos_embed - pos_sim).pow(2) + (neg_embed - neg_sim).pow(2) + (pos_neg_embed - pos_neg_sim).pow(2)
    return (rank + torch.sqrt(mse + 1e-8)).mean()


@dataclass
class TrainResult:
    model: StringEmbedder
    losses: list[float]
    dataset: TripletWords
    device: torch.device
    parameter_count: int
    training_seconds: float

    @property
    def final_loss(self) -> float:
        return self.losses[-1] if self.losses else float("nan")


def save_checkpoint(result: TrainResult, path: str | Path, similarity_fn: SimilarityFn = name_similarity) -> Path:
    path = Path(path)
    checkpoint = {
        "model_state_dict": {key: value.cpu() for key, value in result.model.state_dict().items()},
        "model_args": {
            "max_len": result.model.max_len,
            "embed_dim": result.model.embed_dim,
            "char_dim": result.model.char_dim,
            "channels": result.model.channels,
            "kernels": result.model.kernels,
            "max_tokens": result.model.max_tokens,
            "token_len": result.model.token_len,
            "token_dim": result.model.token_dim,
            "token_channels": result.model.token_channels,
            "token_kernels": result.model.token_kernels,
            "projection_dim": result.model.projection_dim,
            "dropout": result.model.dropout,
        },
        "losses": result.losses,
        "device": result.device.type,
        "similarity_fn": similarity_fn.__name__,
        "word_count": len(result.dataset.words),
        "parameter_count": result.parameter_count,
        "training_seconds": result.training_seconds,
        "epochs": len(result.losses),
    }
    torch.save(checkpoint, path)
    return path


def train(
    words: list[str],
    epochs: int = 5,
    batch_size: int = 256,
    lr: float = 1e-3,
    progress: bool = False,
    device: str | torch.device | None = None,
    output_path: str | Path | None = None,
    resume_from: str | Path | None = None,
    context_chars: int = NAME_CONTEXT_CHARS,
    k: int = 8,
    sample_size: int = 64,
    embed_dim: int = 128,
    char_dim: int = 32,
    channels: int = 36,
    kernels: tuple[int, ...] = (2, 3, 4, 5),
    max_tokens: int = 8,
    token_len: int = 24,
    token_dim: int = 64,
    token_channels: int = 48,
    token_kernels: tuple[int, ...] = (1, 2, 3),
    projection_dim: int = 128,
    dropout: float = 0.1,
    num_workers: int = 0,
    compile_model: bool = False,
    similarity_fn: SimilarityFn = name_similarity,
) -> TrainResult:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = TripletWords(
        words,
        max_len=context_chars,
        k=k,
        sample_size=sample_size,
        similarity_fn=similarity_fn,
        progress=progress,
    )
    model = StringEmbedder(
        max_len=dataset.max_len,
        embed_dim=embed_dim,
        char_dim=char_dim,
        channels=channels,
        kernels=kernels,
        max_tokens=max_tokens,
        token_len=token_len,
        token_dim=token_dim,
        token_channels=token_channels,
        token_kernels=token_kernels,
        projection_dim=projection_dim,
        dropout=dropout,
    ).to(device)
    parameter_count = count_parameters(model)
    if parameter_count > MODEL_PARAMETER_LIMIT:
        raise ValueError(f"Model has {parameter_count:,} parameters, above the {MODEL_PARAMETER_LIMIT:,} parameter limit.")
    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    train_model = torch.compile(model) if compile_model and device.type == "cuda" else model

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    losses: list[float] = []
    epoch_iter = _progress(range(epochs), progress, desc="training tiny embedder", unit="epoch")

    model.train()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_started = time.perf_counter()
    for _ in epoch_iter:
        total = 0.0
        for anchor, positive, negative, pos_sim, neg_sim, pos_neg_sim in loader:
            anchor = _move_encoded(anchor, device)
            positive = _move_encoded(positive, device)
            negative = _move_encoded(negative, device)
            pos_sim = pos_sim.to(device)
            neg_sim = neg_sim.to(device)
            pos_neg_sim = pos_neg_sim.to(device)

            optimizer.zero_grad()
            loss = triplet_loss((train_model(anchor), train_model(positive), train_model(negative)), (pos_sim, neg_sim, pos_neg_sim))
            loss.backward()
            optimizer.step()
            total += loss.item()

        epoch_loss = total / max(len(loader), 1)
        losses.append(epoch_loss)
        if progress and hasattr(epoch_iter, "set_postfix"):
            cast(Any, epoch_iter).set_postfix(
                device=device.type,
                loss=f"{epoch_loss:.4f}",
                lr=f"{lr:.2e}",
                params=f"{parameter_count:,}",
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - train_started

    result = TrainResult(
        model=model,
        losses=losses,
        dataset=dataset,
        device=device,
        parameter_count=parameter_count,
        training_seconds=training_seconds,
    )
    if output_path is not None:
        save_checkpoint(result, output_path, similarity_fn=similarity_fn)
    return result


def nearest(model: StringEmbedder, query: str, words: list[str], n: int = DEFAULT_CANDIDATE_COUNT):
    return [
        (word, embedding_distance, normalized_edit_distance(query, word))
        for word, embedding_distance in nearest_by_embedding(model, query, words, n)
    ]


def _move_encoded(encoded, device: torch.device):
    return tuple(part.to(device) for part in encoded)
