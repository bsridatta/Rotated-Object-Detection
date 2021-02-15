import torch
from src.rotated_ship_data import _make_box_pts
from shapely.geometry import Polygon
import numpy as np


def compute_metrics(pred, target, iou_threshold=0.7, pr_score=0.5):
    """Compute IOU, AP, Precision, Recall, F1 score

    Arguments:
        pred {Tensor Batch} -- b, [p(ship), x, y, yaw, w, h]
        target {Tensor Batch} -- b, [p(ship), x, y, yaw, w, h]

    Keyword Arguments:
        iou_threshold {float} -- predicted bbox is correct if IOU > this value (default: {0.7})
        pr_score {float} -- object conf. threshold to sample precision and recall (default: {0.5})

    Returns:
        precision, recall, F1 @ pr_score, AP@ iou_threshold and mean IOU

    Reference: 
        https://github.com/ultralytics/yolov3/blob/e0a5a6b411cca45f0d64aa932abffbf3c99b92b3/test.py
    """
    pred = pred.detach()
    target = target.detach()
    conf = pred[:, 0]

    # TPs based on IOU threshold
    ious = torch.zeros((pred.shape[0], 1), dtype=float, device=pred.device)
    tp = torch.zeros((pred.shape[0], 1), dtype=bool, device=pred.device)

    # get all IOUS if there is bbox in target
    for i in torch.nonzero(target[:, 0], as_tuple=False):
        # enable convertion to numpy in _make_box_pts
        t = Polygon(_make_box_pts(*target[i, 1:].flatten().cpu()))
        p = Polygon(_make_box_pts(*pred[i, 1:].flatten().cpu()))
        iou = t.intersection(p).area / t.union(p).area
        ious[i] = iou

    # is TP if IOU > threshold
    tp[ious > iou_threshold] = True

    mean_iou = torch.mean(ious)

    # Calcualted Precision, Recall, F1, AP
    # sort by conf
    sorted_idx = torch.argsort(conf, dim=0, descending=True)
    tp, conf = tp[sorted_idx], conf[sorted_idx]

    tp = tp * 1.0  # boolean to float
    # TP, FP Cummulative
    tpc = torch.cumsum(tp, dim=0)
    fpc = torch.cumsum(1-tp, dim=0)

    # TP + FN = N(Target=1) constant
    eps = 1e-20
    sum_tp_fn = (target[:, 0]).sum() + eps
    prec = tpc / (tpc + fpc)
    rec = tpc / sum_tp_fn

    # One P, R at conf threshold
    # -1 as conf decreases along x
    p = torch.tensor(np.interp(-pr_score, -conf.cpu(), prec[:, 0].cpu()))
    r = torch.tensor(np.interp(-pr_score, -conf.cpu(), rec[:, 0].cpu()))

    ap = compute_ap(list(rec), list(prec))

    f1 = 2 * p * r / (p + r + eps)

    return p, r, f1, ap, mean_iou


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.

    Code Source: 
        unmodified - https://github.com/rbgirshick/py-faster-rcnn.

    Reference: 
        https://github.com/ultralytics/yolov3/blob/e0a5a6b411cca45f0d64aa932abffbf3c99b92b3/test.py

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(
        ([0.], recall, [min(recall[-1] + 1E-3, 1.)])).astype('float')
    mpre = np.concatenate(([0.], precision, [0.]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre))).astype('float')

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        # points where x axis (recall) changes
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap
