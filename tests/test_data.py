import pytest
from src.dataset import Ships
import torch


@pytest.fixture
def random():
    torch.manual_seed(0)


def test_label_has_ship():
    dataset = Ships(15)
    for i in range(len(dataset)):
        sample = dataset[i]['target']
        if sample[0] == 1:
            assert ~torch.isnan(sample[1]).item()
        elif sample[0] == 0:
            assert torch.isnan(sample[1]).item()
        else:
            assert False
