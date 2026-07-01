import numpy as np
import torch

from string_embed import StringEmbedder, count_parameters, encode_text, nearest_by_embedding, normalize_name


def test_normalize_name_keeps_ascii_tokens_and_digits():
    assert normalize_name("  José A. Smith-2  ") == "jose a smith 2"


def test_encode_text_shapes_match_model_defaults():
    raw_ids, token_ids, token_mask = encode_text("John A Smith")

    assert raw_ids.shape == (32,)
    assert token_ids.shape == (8, 24)
    assert token_mask.tolist()[:3] == [True, True, True]


def test_model_outputs_unit_embeddings_under_parameter_limit():
    model = StringEmbedder()
    raw_ids, token_ids, token_mask = zip(*(encode_text(text) for text in ["John Smith", "Jon Smith"]), strict=True)

    embeddings = model((torch.stack(raw_ids), torch.stack(token_ids), torch.stack(token_mask)))

    assert embeddings.shape == (2, 128)
    assert count_parameters(model) == 200_544
    assert count_parameters(model) < 1_000_000
    np.testing.assert_allclose(embeddings.norm(dim=1).detach().numpy(), np.ones(2), rtol=1e-5)


def test_nearest_by_embedding_returns_requested_count():
    model = StringEmbedder()
    words = ["john smith", "jane baker", "robert jones"]

    results = nearest_by_embedding(model, "john smoth", words, n=2)

    assert len(results) == 2
    assert all(isinstance(word, str) and isinstance(distance, float) for word, distance in results)
