import torch

from string_embed.train import TripletWords, name_similarity, normalized_edit_distance, train


def test_name_similarity_rewards_token_reordering():
    reordered = name_similarity("John A Smith", "Smith John A")
    unrelated = name_similarity("John A Smith", "Martha Baker")

    assert reordered > unrelated


def test_normalized_edit_distance_bounds():
    assert normalized_edit_distance("abc", "abc") == 0
    assert normalized_edit_distance("abc", "xyz") == 1


def test_train_runs_one_tiny_epoch():
    words = ["John Smith", "Jon Smith", "Jane Baker", "Robert Jones", "Rob Jones", "Alice Stone"]

    result = train(words, epochs=1, batch_size=3, sample_size=4, device="cpu")

    assert result.parameter_count == 200_544
    assert len(result.losses) == 1
    assert result.final_loss >= 0
    assert result.training_seconds > 0
    assert isinstance(result.dataset, TripletWords)
    assert next(result.model.parameters()).device == torch.device("cpu")
