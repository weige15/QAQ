import pytest
import torch

from qaq.router.features import masked_mean_pool, validate_prompt_mask


def test_masked_mean_pool_is_explicit_sum_over_valid_positions():
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 5.0], [100.0, 200.0]]])
    mask = torch.tensor([[1, 1, 0]])
    assert torch.equal(masked_mean_pool(hidden, mask), torch.tensor([2.0, 3.5]))


def test_pooling_rejects_missing_invalid_and_all_padding_masks():
    hidden = torch.zeros(1, 3, 2)
    with pytest.raises(ValueError, match="explicit attention_mask"):
        validate_prompt_mask(None, sequence_length=3)
    with pytest.raises(ValueError, match="only 0/1"):
        validate_prompt_mask(torch.tensor([[1, 2, 0]]), sequence_length=3)
    with pytest.raises(ValueError, match="at least one"):
        masked_mean_pool(hidden, torch.zeros(1, 3, dtype=torch.long))


def test_pooling_requires_batch_size_one_and_matching_sequence_length():
    with pytest.raises(ValueError, match="batch-size-one"):
        masked_mean_pool(torch.zeros(2, 3, 4), torch.ones(2, 3))
    with pytest.raises(ValueError, match="sequence length"):
        validate_prompt_mask(torch.ones(1, 2), sequence_length=3)
