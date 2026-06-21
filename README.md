# String Embed

Train a small convolutional model that embeds strings into vectors whose cosine distance roughly tracks normalized edit distance.

## Install

This repo uses [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

For notebook work, install the development dependencies too:

```bash
uv sync --group dev
```

## Train

Run a short training job against the bundled word list:

```bash
uv run python - <<'PY'
import torch

from testdata import load_words
from train import train

result = train(load_words(limit=256), epochs=5)
print(result.losses)
torch.save({"model_state_dict": result.model.state_dict()}, "string_embed.pt")
PY
```

## Query

After saving a checkpoint, query nearest words by embedding distance:

```bash
uv run string-embed string_embed.pt apple --top 5
```

Use your own newline-delimited word list with `--words`:

```bash
uv run string-embed string_embed.pt apple --words words.txt --limit 10000 --top 10
```

## Dev Container

The dev container includes uv and runs `uv sync --group dev` after creation so the virtual environment is ready when the container starts. If dependencies change, run the same command again.
