import os
from argparse import ArgumentParser

import numpy as np
import torch

import src.dataloader as loader
import src.models as models
from src.metrics import compute_metrics
from src.trainer import validation_epoch


def main():
    # Experiment configuration, opt, is distributed to all the other modules
    opt = _do_setup()

    test_loader = loader.test_dataloader(opt)

    model = models.Detector_FPN()
    model.to(opt.device)
    state = torch.load(
        f"{os.path.dirname(os.path.abspath(__file__))}/checkpoints/model_93_ap.pt",
        map_location=opt.device,
    )
    model.load_state_dict(state["model_state_dict"])

    # snippet from src/trainer.py/validation_epoch()
    model.eval()

    ap = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            for key in batch.keys():
                batch[key] = batch[key].to(opt.device)

            # validation step
            input, target = batch["input"], batch["target"]
            output = model(input)

            _prec, _rec, _f1, _ap, _iou = compute_metrics(output, target)
            ap.append(_ap)

    avg_ap = sum(ap) / len(ap)

    print(f"\n AP on {len(test_loader.dataset)} samples: {avg_ap}")


def _do_setup():
    parser = _get_argparser()
    opt = parser.parse_args()

    # fix seed for reproducibility
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    # GPU setup
    use_cuda = opt.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    opt.device = device  # Adding device to opt, not already in argparse
    opt.num_workers = 4 if use_cuda else 4  # to tune per device
    return opt


def _get_argparser():

    parser = ArgumentParser()
    # training specific
    # fmt: off
    parser.add_argument("--batch_size", default=256, type=int,
                        help="number of samples per step, have more than one for batch norm")
    parser.add_argument("--resume_run", default="None", type=str,
                        help="auto load ckpt")
    # data
    parser.add_argument("--test_len", default=8000, type=int,
                        help="number of samples for testing")
    # device
    parser.add_argument("--cuda", default=True, type=lambda x: (str(x).lower() == "true"),
                        help="enable cuda if available")
    parser.add_argument("--pin_memory", default=False, type=lambda x: (str(x).lower() == "true"),
                        help="pin memory to device")
    parser.add_argument("--seed", default=400, type=int,
                        help="random seed")
    # fmt: on
    return parser


if __name__ == "__main__":
    main()
