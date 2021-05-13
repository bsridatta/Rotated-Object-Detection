import gc
from argparse import Namespace

import torch
from torch.nn.modules.module import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.callbacks.base import CallbackList
from src.loss import compute_loss
from src.metrics import compute_metrics
from src.rotated_ship_data import score_iou


def training_epoch(
    cb: CallbackList,
    opt: Namespace,
    model: Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
) -> None:
    """logic for each training epoch"""
    model.train()

    for batch_idx, batch in enumerate(train_loader):
        for key in batch.keys():
            batch[key] = batch[key].to(opt.device)

        optimizer.zero_grad()

        # training step
        input, target = batch["input"], batch["target"]
        output = model(input)
        loss, _l_ship, _l_bbox = compute_loss(output, target)
        loss = loss.mean()
        loss.backward()
        optimizer.step()

        # required info for - logging cb
        cb.on_train_batch_end(
            opt=opt,
            batch_idx=batch_idx,
            batch=batch,
            dataloader=train_loader,
            output=loss.item(),
            l_ship=_l_ship.mean().item(),
            l_bbox=_l_bbox.mean().item(),
        )

        del loss
        del batch
        gc.collect()


def validation_epoch(
    cb: CallbackList,
    opt: Namespace,
    model: Module,
    val_loader: DataLoader,
) -> torch.Tensor:
    """logic for each validation epoch"""
    model.eval()

    # metrics to return
    losses = []
    prec = []
    rec = []
    f1 = []
    ap = []
    iou = []
    l_ship = []
    l_bbox = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            for key in batch.keys():
                batch[key] = batch[key].to(opt.device)

            # validation step
            input, target = batch["input"], batch["target"]
            output = model(input)

            loss, _l_ship, _l_bbox = compute_loss(output, target)
            _prec, _rec, _f1, _ap, _iou = compute_metrics(output, target)

            # append incase analysis of distribution is of interest
            losses.append(loss)
            l_ship.append(_l_ship)
            l_bbox.append(_l_bbox)
            prec.append(_prec)
            rec.append(_rec)
            f1.append(_f1)
            ap.append(_ap)
            iou.append(_iou)

    loss_avg = torch.mean(torch.cat(losses))
    l_ship = torch.mean(torch.cat(l_ship))
    l_bbox = torch.mean(torch.cat(l_bbox))

    metrics = {}
    for k, m in zip(["prec", "rec", "f1", "ap", "iou"], [prec, rec, f1, ap, iou]):
        m = sum(m) / len(m)
        metrics[k] = m

    cb.on_validation_end(
        opt=opt, output=loss_avg, metrics=metrics, l_ship=l_ship, l_bbox=l_bbox
    )

    return loss_avg
