from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ALPHABET = "abcdefghijklmnopqrstuvwxyz"
CHAR_TO_ID = {c: i for i, c in enumerate(ALPHABET)}


def encode_word(word: str, max_len: int) -> torch.Tensor:
    x = torch.zeros(len(ALPHABET), max_len, dtype=torch.float32)
    for pos, char in enumerate(word.lower()[:max_len]):
        if char in CHAR_TO_ID:
            x[CHAR_TO_ID[char], pos] = 1.0
    return x


class StringEmbedder(nn.Module):
    def __init__(self, max_len: int, embed_dim: int = 32, channels: int = 8):
        super().__init__()
        self.max_len = max_len
        self.net = nn.Sequential(
            nn.Conv1d(len(ALPHABET), channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(channels * (max_len // 2), embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=-1)

    @torch.no_grad()
    def embed_words(self, words: list[str], device: str | torch.device = "cpu") -> np.ndarray:
        self.eval()
        x = torch.stack([encode_word(word, self.max_len) for word in words]).to(device)
        return self.to(device)(x).cpu().numpy()


def nearest_by_embedding(model: StringEmbedder, query: str, words: list[str], n: int = 5):
    embeddings = model.embed_words([query] + words)
    scores = 1 - embeddings[1:] @ embeddings[0]
    return [(words[i], float(scores[i])) for i in np.argsort(scores)[:n]]
