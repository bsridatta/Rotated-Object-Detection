import pytest
import torch

from src.loss import compute_loss, lmr5p
from src.metrics import compute_metrics


@pytest.fixture
def random():
    torch.manual_seed(0)


nan = float("nan")


# pred, target
# True, True    TP # IOU = 1
# False, False  TN # IOU = 1
# False, True   FN # IOU = 1
# True, False   FP # IOU = 1
# True, True    FP # IOU = 0 TP -> FP IOU<Threshold
# False, False  TN # IOU = 0
# False, True   FN # IOU = 0
# True, False   FP # IOU = 0


@pytest.mark.parametrize(
    "pred, target",
    [
        ([1, 0.6, 0.5, 0.4, 0.3, 0.2], [1, 0.6, 0.5, 0.4, 0.3, 0.2]),
        ([0.2, 0.6, 0.5, 0.4, 0.3, 0.2], [0, nan, nan, nan, nan, nan]),
        ([0.1, 0.6, 0.5, 0.4, 0.3, 0.2], [1, 0.6, 0.5, 0.4, 0.3, 0.2]),
        ([0.9, 0.6, 0.5, 0.4, 0.3, 0.2], [0, nan, nan, nan, nan, nan]),
        ([0.8, 10.2, 10.3, 10.4, 10.5, 10.6], [1, 0.6, 0.5, 0.4, 0.3, 0.2]),
        ([0.2, 10.2, 10.3, 10.4, 10.5, 10.6], [0, nan, nan, nan, nan, nan]),
        ([0.5, 10.2, 10.3, 10.4, 10.5, 10.6], [1, 0.6, 0.5, 0.4, 0.3, 0.2]),
        ([0.9, 10.2, 10.3, 10.4, 10.5, 10.6], [0, nan, nan, nan, nan, nan]),
    ],
)
def test_compute_metrics(pred, target):
    pred = torch.tensor([pred], requires_grad=True)
    target = torch.tensor([target], requires_grad=True)

    prec, rec, f1, ap, iou = compute_metrics(pred, target, pr_score=0.5)
    assert isinstance(prec, torch.Tensor)
    # assert tp[0] == True and tp[4] == False


def test_lmr5():
    pred = torch.rand(3, 5)
    target = torch.rand(3, 5)
    loss = lmr5p(pred, target)

    assert len(loss) == len(pred)
    assert isinstance(loss, torch.Tensor)


def test_compute_loss():
    nan = float("nan")

    pred = torch.Tensor([[0, 0.2, 0.3, 0.4, 0.5, 0.6], [1, 0.2, 0.3, 0.4, 0.5, 0.6]])
    target = torch.Tensor([[0, nan, nan, nan, nan, nan], [1, 0.6, 0.5, 0.4, 0.3, 0.2]])

    loss, l_ship, l_bbox = compute_loss(pred, target)

    assert loss is not nan
