from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model import StringEmbedder, nearest_by_embedding
from testdata import load_words


def _state_dict_from_checkpoint(checkpoint):
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        if "net.0.weight" in checkpoint:
            return checkpoint
    return None


def _model_from_state_dict(state_dict) -> StringEmbedder:
    conv = state_dict["net.0.weight"]
    linear_key = next(key for key in state_dict if key.startswith("net.") and key.endswith(".weight") and key != "net.0.weight")
    linear = state_dict[linear_key]
    channels = conv.shape[0]
    embed_dim, flat_size = linear.shape
    max_len = flat_size // channels * 2
    model = StringEmbedder(max_len=max_len, embed_dim=embed_dim, channels=channels)
    model.load_state_dict(state_dict)
    return model


def load_model(path: str | Path, device: str = "cpu") -> StringEmbedder:
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, StringEmbedder):
        return checkpoint.to(device).eval()

    state_dict = _state_dict_from_checkpoint(checkpoint)
    if state_dict is None:
        raise ValueError("Expected a StringEmbedder, state_dict, or checkpoint with model_state_dict/state_dict.")

    return _model_from_state_dict(state_dict).to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description="Load a string embedding model and query nearest words.")
    parser.add_argument("model", help="Path to a saved StringEmbedder or state_dict checkpoint.")
    parser.add_argument("query", help="String to embed and search for.")
    parser.add_argument("--words", help="Optional newline-delimited word file. Defaults to bundled test data.")
    parser.add_argument("--limit", type=int, default=256, help="Maximum words to load.")
    parser.add_argument("--top", type=int, default=5, help="Number of nearest words to print.")
    parser.add_argument("--device", default="cpu", help="Torch device, for example cpu or cuda.")
    args = parser.parse_args()

    model = load_model(args.model, device=args.device)
    words = load_words(args.words, limit=args.limit)

    for word, distance in nearest_by_embedding(model, args.query, words, n=args.top):
        print(f"{word}\t{distance:.6f}")


if __name__ == "__main__":
    main()
