from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch

from string_embed import StringEmbedder
from string_embed.testdata import load_words


def mutate(text: str, rng: random.Random) -> str:
    text = text.lower()
    variants: list[str] = []
    if len(text) > 4:
        index = rng.randrange(len(text))
        variants.append(text[:index] + text[index + 1 :])
    if len(text) > 4:
        index = rng.randrange(len(text) - 1)
        variants.append(text[:index] + text[index + 1] + text[index] + text[index + 2 :])
    if text:
        index = rng.randrange(len(text))
        variants.append(text[:index] + rng.choice("abcdefghijklmnopqrstuvwxyz") + text[index + 1 :])
    if len(text) > 3:
        index = rng.randrange(1, len(text))
        variants.append(text[:index] + " " + text[index:])
    return rng.choice([variant for variant in variants if variant and variant != text] or [text])


def recall_at(ranks: np.ndarray, k: int) -> float:
    return float(np.mean(ranks <= k))


def benchmark(
    checkpoint_path: Path,
    word_count: int,
    query_count: int,
    seed: int,
    batch_size: int,
) -> dict[str, float]:
    rng = random.Random(seed)
    words = load_words(limit=word_count, seed=13)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = StringEmbedder(**checkpoint["model_args"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    originals = rng.sample(words, min(query_count, len(words)))
    queries = [mutate(word, rng) for word in originals]

    start = time.perf_counter()
    database_embeddings = model.embed_words(words, batch_size=batch_size)
    database_seconds = time.perf_counter() - start

    start = time.perf_counter()
    query_embeddings = model.embed_words(queries, batch_size=batch_size)
    query_seconds = time.perf_counter() - start

    scores = 1 - query_embeddings @ database_embeddings.T
    order = np.argsort(scores, axis=1)
    word_to_index = {word: index for index, word in enumerate(words)}
    ranks = np.array(
        [int(np.where(row == word_to_index[word])[0][0]) + 1 for word, row in zip(originals, order, strict=True)]
    )

    return {
        "recall@1": recall_at(ranks, 1),
        "recall@5": recall_at(ranks, 5),
        "recall@10": recall_at(ranks, 10),
        "recall@25": recall_at(ranks, 25),
        "recall@50": recall_at(ranks, 50),
        "recall@100": recall_at(ranks, 100),
        "median_rank": float(np.median(ranks)),
        "p95_rank": float(np.percentile(ranks, 95)),
        "mean_rank": float(np.mean(ranks)),
        "database_embed_seconds": database_seconds,
        "query_embed_seconds": query_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate mutation recall for a string-embed checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=Path("tiny_string_embed.pt"))
    parser.add_argument("--word-count", type=int, default=10_000)
    parser.add_argument("--query-count", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()

    metrics = benchmark(args.checkpoint, args.word_count, args.query_count, args.seed, args.batch_size)
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}" if key.startswith("recall") else f"{key}: {value:.3f}")


if __name__ == "__main__":
    main()
