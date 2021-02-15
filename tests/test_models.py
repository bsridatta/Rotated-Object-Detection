import pytest
import torch
from src import models


@pytest.fixture
def random():
    torch.manual_seed(0)


def test_models():

    inp = torch.rand((2, 1, 200, 200))

    model = models.Detector()
    out = model(inp)
    assert out.shape == (inp.shape[0], 6)

    model = models.Detector_FPN()
    out = model(inp)
    assert out.shape == (inp.shape[0], 6)
