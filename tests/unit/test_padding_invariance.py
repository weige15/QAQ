import torch

from qaq.router.features import masked_mean_pool


def test_right_padding_does_not_change_prompt_feature():
    content = torch.tensor([[[1.0, 2.0], [3.0, 5.0]]])
    padded = torch.cat((content, torch.tensor([[[99.0, 101.0], [77.0, 88.0]]])), dim=1)
    short = masked_mean_pool(content, torch.tensor([[1, 1]]))
    long = masked_mean_pool(padded, torch.tensor([[1, 1, 0, 0]]))
    assert torch.allclose(short, long, atol=1e-6, rtol=0.0)
