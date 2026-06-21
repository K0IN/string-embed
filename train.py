from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model import StringEmbedder, encode_word, nearest_by_embedding


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


SimilarityFn = Callable[[str, str], float]


def similarity_matrix(words: list[str], similarity_fn: SimilarityFn = normalized_levenshtein_similarity) -> np.ndarray:
    similarities = np.eye(len(words), dtype=np.float32)
    for i, a in enumerate(words):
        for j in range(i + 1, len(words)):
            similarities[i, j] = similarities[j, i] = similarity_fn(a, words[j])
    return similarities


class TripletWords(Dataset):
    def __init__(
        self,
        words: list[str],
        max_len: int | None = None,
        k: int = 8,
        similarity_fn: SimilarityFn = normalized_levenshtein_similarity,
    ):
        if len(words) < 2:
            raise ValueError("Need at least two words to train triplets.")
        self.words = [word.lower() for word in words]
        self.max_len = max(2, max_len or max(map(len, self.words)))
        if self.max_len % 2:
            self.max_len += 1
        self.similarities = similarity_matrix(self.words, similarity_fn)
        self.neighbors = np.argsort(-self.similarities, axis=1)
        self.k = min(k, max(1, len(self.words) - 1))

    def __len__(self) -> int:
        return len(self.words)

    def __getitem__(self, idx: int):
        hi = min(len(self.words) - 1, self.k * 2)
        pos = int(self.neighbors[idx, random.randint(1, hi)])
        neg = int(self.neighbors[idx, random.randint(1, hi)])
        if self.similarities[idx, pos] < self.similarities[idx, neg]:
            pos, neg = neg, pos
        return (
            encode_word(self.words[idx], self.max_len),
            encode_word(self.words[pos], self.max_len),
            encode_word(self.words[neg], self.max_len),
            torch.tensor(self.similarities[idx, pos], dtype=torch.float32),
            torch.tensor(self.similarities[idx, neg], dtype=torch.float32),
            torch.tensor(self.similarities[pos, neg], dtype=torch.float32),
        )


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


def train(
    words: list[str],
    epochs: int = 5,
    batch_size: int = 32,
    embed_dim: int = 32,
    lr: float = 1e-3,
    progress: bool = False,
    device: str | torch.device | None = None,
    similarity_fn: SimilarityFn = normalized_levenshtein_similarity,
) -> TrainResult:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = TripletWords(words, similarity_fn=similarity_fn)
    model = StringEmbedder(dataset.max_len, embed_dim=embed_dim).to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses: list[float] = []
    epoch_iter = range(epochs)
    if progress:
        from tqdm.auto import tqdm

        epoch_iter = tqdm(epoch_iter, desc="Training", unit="epoch")

    model.train()
    for _ in epoch_iter:
        total = 0.0
        for anchor, positive, negative, pos_sim, neg_sim, pos_neg_sim in loader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            pos_sim = pos_sim.to(device)
            neg_sim = neg_sim.to(device)
            pos_neg_sim = pos_neg_sim.to(device)
            optimizer.zero_grad()
            loss = triplet_loss((model(anchor), model(positive), model(negative)), (pos_sim, neg_sim, pos_neg_sim))
            loss.backward()
            optimizer.step()
            total += loss.item()
        epoch_loss = total / max(len(loader), 1)
        losses.append(epoch_loss)
        if progress:
            epoch_iter.set_postfix(device=device.type, loss=f"{epoch_loss:.4f}")
    return TrainResult(model=model, losses=losses, dataset=dataset, device=device)


def nearest(model: StringEmbedder, query: str, words: list[str], n: int = 5):
    return [
        (word, embedding_distance, normalized_edit_distance(query, word))
        for word, embedding_distance in nearest_by_embedding(model, query, words, n)
    ]
