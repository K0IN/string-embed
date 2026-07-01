"""Tiny string embeddings for dedupe candidate preselection."""

from string_embed.model import (
    DEFAULT_CANDIDATE_COUNT,
    MODEL_PARAMETER_LIMIT,
    NAME_CONTEXT_CHARS,
    StringEmbedder,
    count_parameters,
    encode_text,
    encode_word,
    nearest_by_embedding,
    normalize_name,
)
from string_embed.train import (
    TrainResult,
    TripletWords,
    edit_distance,
    name_similarity,
    nearest,
    normalized_edit_distance,
    normalized_levenshtein_similarity,
    save_checkpoint,
    train,
    triplet_loss,
)

__all__ = [
    "DEFAULT_CANDIDATE_COUNT",
    "MODEL_PARAMETER_LIMIT",
    "NAME_CONTEXT_CHARS",
    "StringEmbedder",
    "TrainResult",
    "TripletWords",
    "count_parameters",
    "edit_distance",
    "encode_text",
    "encode_word",
    "name_similarity",
    "nearest",
    "nearest_by_embedding",
    "normalize_name",
    "normalized_edit_distance",
    "normalized_levenshtein_similarity",
    "save_checkpoint",
    "train",
    "triplet_loss",
]
