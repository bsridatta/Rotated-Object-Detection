import pytest
from src.metrics import compute_metrics
from src.loss import lmr5p, compute_loss
import torch


@pytest.fixture
def random():
    torch.manual_seed(0)


def test_compute_metrics():
    nan = float('nan')

    # pred, target
    # True, True    TP # IOU = 1
    # False, False  TN # IOU = 1
    # False, True   FN # IOU = 1
    # True, False   FP # IOU = 1
    # True, True    FP # IOU = 0 TP -> FP IOU<Threshold
    # False, False  TN # IOU = 0
    # False, True   FN # IOU = 0
    # True, False   FP # IOU = 0
    pred = torch.tensor([[1, 0.6, 0.5, 0.4, 0.3, 0.2],
                         [0.2, 0.6, 0.5, 0.4, 0.3, 0.2],
                         [0.1, 0.6, 0.5, 0.4, 0.3, 0.2],
                         [0.9, 0.6, 0.5, 0.4, 0.3, 0.2],
                         [0.8, 10.2, 10.3, 10.4, 10.5, 10.6],
                         [0.2, 10.2, 10.3, 10.4, 10.5, 10.6],
                         [0.5, 10.2, 10.3, 10.4, 10.5, 10.6],
                         [0.9, 10.2, 10.3, 10.4, 10.5, 10.6]
                         ], requires_grad=True)

    target = torch.tensor([[1, 0.6, 0.5, 0.4, 0.3, 0.2],
                           [0, nan, nan, nan, nan, nan],
                           [1, 0.6, 0.5, 0.4, 0.3, 0.2],
                           [0, nan, nan, nan, nan, nan],
                           [1, 0.6, 0.5, 0.4, 0.3, 0.2],
                           [0, nan, nan, nan, nan, nan],
                           [1, 0.6, 0.5, 0.4, 0.3, 0.2],
                           [0, nan, nan, nan, nan, nan]
                           ], requires_grad=True)

    prec, rec, f1, ap, iou = compute_metrics(pred, target, pr_score=0.5)
    assert isinstance (prec, torch.Tensor)
    # assert tp[0] == True and tp[4] == False


def test_lmr5():
    pred = torch.rand(3, 5)
    target = torch.rand(3, 5)
    loss = lmr5p(pred, target)

    assert len(loss) == len(pred)
    assert isinstance(loss, torch.Tensor)

def test_compute_loss():
    nan = float('nan')

    pred = torch.Tensor([[0, 0.2, 0.3, 0.4, 0.5, 0.6],
                         [1, 0.2, 0.3, 0.4, 0.5, 0.6]])
    target = torch.Tensor([[0, nan, nan, nan, nan, nan],
                           [1, 0.6, 0.5, 0.4, 0.3, 0.2]])

    loss, l_ship, l_bbox = compute_loss(pred, target)

    assert (loss is not nan)
