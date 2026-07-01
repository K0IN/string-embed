from __future__ import annotations

import unicodedata

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


MODEL_PARAMETER_LIMIT = 1_000_000
DEFAULT_CANDIDATE_COUNT = 100
NAME_CONTEXT_CHARS = 32
MAX_TOKENS = 8
TOKEN_CONTEXT_CHARS = 24
ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789 "
PAD_ID = 0
UNK_ID = 1
CHAR_TO_ID = {c: i + 2 for i, c in enumerate(ALPHABET)}
VOCAB_SIZE = len(ALPHABET) + 2


def normalize_name(name: str) -> str:
    decomposed = unicodedata.normalize("NFKD", name)
    ascii_name = "".join(char for char in decomposed if not unicodedata.combining(char))
    chars: list[str] = []
    last_was_space = True
    for char in ascii_name.lower():
        if char.isalnum():
            chars.append(char if char in CHAR_TO_ID else " ")
            last_was_space = False
        elif not last_was_space:
            chars.append(" ")
            last_was_space = True
    return "".join(chars).strip()


def encode_word(word: str, max_len: int) -> torch.Tensor:
    x = torch.zeros(max_len, dtype=torch.long)
    for pos, char in enumerate(normalize_name(word)[:max_len]):
        x[pos] = CHAR_TO_ID.get(char, UNK_ID)
    return x


def encode_text(
    text: str,
    max_len: int = NAME_CONTEXT_CHARS,
    max_tokens: int = MAX_TOKENS,
    token_len: int = TOKEN_CONTEXT_CHARS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    normalized = normalize_name(text)[:max_len]
    raw_ids = encode_word(normalized, max_len)
    token_ids = torch.zeros(max_tokens, token_len, dtype=torch.long)
    token_mask = torch.zeros(max_tokens, dtype=torch.bool)
    for row, token in enumerate(normalized.split()[:max_tokens]):
        token_ids[row] = encode_word(token, token_len)
        token_mask[row] = True
    return raw_ids, token_ids, token_mask


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


class StringEmbedder(nn.Module):
    def __init__(
        self,
        max_len: int = NAME_CONTEXT_CHARS,
        embed_dim: int = 128,
        char_dim: int = 32,
        channels: int = 36,
        kernels: tuple[int, ...] = (2, 3, 4, 5),
        max_tokens: int = MAX_TOKENS,
        token_len: int = TOKEN_CONTEXT_CHARS,
        token_dim: int = 64,
        token_channels: int = 48,
        token_kernels: tuple[int, ...] = (1, 2, 3),
        projection_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_len = max(max_len, max(kernels))
        self.max_tokens = max(max_tokens, max(token_kernels))
        self.token_len = max(token_len, max(kernels))
        self.embed_dim = embed_dim
        self.char_dim = char_dim
        self.channels = channels
        self.kernels = tuple(kernels)
        self.token_dim = token_dim
        self.token_channels = token_channels
        self.token_kernels = tuple(token_kernels)
        self.projection_dim = projection_dim
        self.dropout = dropout

        self.char_embedding = nn.Embedding(VOCAB_SIZE, char_dim, padding_idx=PAD_ID)
        self.ngram_convs = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(char_dim, channels, kernel_size=kernel, bias=False),
                nn.GELU(),
                nn.BatchNorm1d(channels),
            )
            for kernel in self.kernels
        )
        sequence_dim = len(self.kernels) * channels * 2
        self.token_projection = nn.Sequential(
            nn.Linear(sequence_dim, token_dim),
            nn.GELU(),
        )
        self.token_order_convs = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(token_dim, token_channels, kernel_size=kernel, bias=False),
                nn.GELU(),
                nn.BatchNorm1d(token_channels),
            )
            for kernel in self.token_kernels
        )
        pooled_dim = sequence_dim * 3 + len(self.token_kernels) * token_channels
        self.projection = nn.Sequential(
            nn.Linear(pooled_dim, projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, embed_dim),
        )

    def forward(self, x) -> torch.Tensor:
        if isinstance(x, (tuple, list)):
            raw_ids, token_ids, token_mask = x
        else:
            raw_ids = self._coerce_raw_ids(x)
            token_ids, token_mask = self._raw_ids_to_single_token(raw_ids)

        raw_ids = self._fit_length(raw_ids, self.max_len)
        token_ids = self._fit_token_ids(token_ids)
        token_mask = self._fit_token_mask(token_mask).to(raw_ids.device)

        raw_features = self._encode_char_sequence(raw_ids)
        batch_size, token_count, token_len = token_ids.shape
        flat_tokens = token_ids.reshape(batch_size * token_count, token_len)
        token_features = self._encode_char_sequence(flat_tokens).reshape(batch_size, token_count, -1)

        mask = token_mask.unsqueeze(-1).to(token_features.dtype)
        token_count_clamped = mask.sum(dim=1).clamp_min(1.0)
        invariant_mean = (token_features * mask).sum(dim=1) / token_count_clamped
        invariant_max = token_features.masked_fill(~token_mask.unsqueeze(-1), -1e9).amax(dim=1)
        has_token = token_mask.any(dim=1, keepdim=True)
        invariant_max = torch.where(has_token, invariant_max, torch.zeros_like(invariant_max))

        ordered_tokens = self.token_projection(token_features).transpose(1, 2)
        ordered_pooled = []
        for conv in self.token_order_convs:
            features = conv(ordered_tokens)
            ordered_pooled.append(features.amax(dim=-1))

        combined = torch.cat([raw_features, invariant_mean, invariant_max, *ordered_pooled], dim=-1)
        return F.normalize(self.projection(combined), p=2, dim=-1)

    def _encode_char_sequence(self, ids: torch.Tensor) -> torch.Tensor:
        embedded = self.char_embedding(ids).transpose(1, 2)
        pooled: list[torch.Tensor] = []
        for conv in self.ngram_convs:
            features = conv(embedded)
            max_pool = features.amax(dim=-1)
            mean_pool = features.mean(dim=-1)
            pooled.extend([max_pool, mean_pool])
        return torch.cat(pooled, dim=-1)

    def _coerce_raw_ids(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.long:
            return self._legacy_one_hot_to_ids(x)
        return x

    def _fit_length(self, x: torch.Tensor, length: int) -> torch.Tensor:
        x = x[:, :length]
        if x.shape[1] < length:
            x = F.pad(x, (0, length - x.shape[1]))
        return x

    def _fit_token_ids(self, token_ids: torch.Tensor) -> torch.Tensor:
        token_ids = token_ids[:, : self.max_tokens, : self.token_len]
        if token_ids.shape[1] < self.max_tokens or token_ids.shape[2] < self.token_len:
            token_ids = F.pad(
                token_ids,
                (0, self.token_len - token_ids.shape[2], 0, self.max_tokens - token_ids.shape[1]),
            )
        return token_ids

    def _fit_token_mask(self, token_mask: torch.Tensor) -> torch.Tensor:
        token_mask = token_mask[:, : self.max_tokens]
        if token_mask.shape[1] < self.max_tokens:
            token_mask = F.pad(token_mask, (0, self.max_tokens - token_mask.shape[1]))
        return token_mask.bool()

    def _raw_ids_to_single_token(self, raw_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        token_ids = torch.zeros(raw_ids.shape[0], self.max_tokens, self.token_len, dtype=torch.long, device=raw_ids.device)
        token_ids[:, 0, : min(self.token_len, raw_ids.shape[1])] = raw_ids[:, : self.token_len]
        token_mask = torch.zeros(raw_ids.shape[0], self.max_tokens, dtype=torch.bool, device=raw_ids.device)
        token_mask[:, 0] = raw_ids.ne(PAD_ID).any(dim=1)
        return token_ids, token_mask

    def _legacy_one_hot_to_ids(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("Expected encoded names with shape [batch, max_len] or legacy one-hot [batch, alphabet, max_len].")
        ids = x.argmax(dim=1) + 2
        empty = x.sum(dim=1) == 0
        ids = ids.masked_fill(empty, PAD_ID)
        return ids.long()

    @torch.no_grad()
    def embed_words(
        self,
        words: list[str],
        device: str | torch.device = "cpu",
        batch_size: int = 4096,
    ) -> np.ndarray:
        self.eval()
        self.to(device)
        embeddings: list[np.ndarray] = []
        for start in range(0, len(words), batch_size):
            batch = words[start : start + batch_size]
            raw_ids, token_ids, token_mask = zip(
                *(encode_text(word, self.max_len, self.max_tokens, self.token_len) for word in batch),
                strict=True,
            )
            x = (
                torch.stack(raw_ids).to(device),
                torch.stack(token_ids).to(device),
                torch.stack(token_mask).to(device),
            )
            embeddings.append(self(x).cpu().numpy())
        return np.concatenate(embeddings, axis=0) if embeddings else np.empty((0, self.embed_dim), dtype=np.float32)


def nearest_by_embedding(model: StringEmbedder, query: str, words: list[str], n: int = DEFAULT_CANDIDATE_COUNT):
    embeddings = model.embed_words([query] + words)
    scores = 1 - embeddings[1:] @ embeddings[0]
    return [(words[i], float(scores[i])) for i in np.argsort(scores)[:n]]
