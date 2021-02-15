import atexit
import os
from argparse import ArgumentParser

import numpy as np
import torch
import gc

import models
import src.dataloader as loader
from src.callbacks import CallbackList, ModelCheckpoint, Logging
from trainer import training_epoch, validation_epoch


def main():
    # Experiment configuration, opt, is distributed to all the other modules
    opt = _do_setup()

    train_loader = loader.train_dataloader(opt)
    val_loader = loader.val_dataloader(opt)
    test_loader = loader.test_dataloader(opt)

    model = models.Detector_FPN()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', verbose=True)

    # data parallel
    if torch.cuda.device_count() > 1:
        print(f'[INFO]: Using {torch.cuda.device_count()} GPUs')
        model = torch.nn.DataParallel(model)

    model.to(opt.device)

    # custom callbacks
    cb = CallbackList([Logging(), ModelCheckpoint()])

    # required info for - checkpoint cb
    cb.setup(opt=opt, model=model, optimizer=optimizer)

    # Train and Val
    for epoch in range(1, opt.epochs+1):
        opt.epoch = epoch
        training_epoch(cb, opt, model, train_loader, optimizer)
        val_loss = validation_epoch(cb, opt, model, val_loader)
        scheduler.step(val_loss)

        # required info for - checkpoint cb
        cb.on_epoch_end(opt=opt, val_loss=val_loss,
                        model=model, optimizer=optimizer, epoch=epoch)

        del val_loss
        gc.collect()

    # sync opt with wandb for easy experiment comparision
    if opt.use_wandb:
        wandb = opt.logger
        opt.logger = None  # wandb cant have objects in its config
        wandb.config.update(opt)


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
    opt.run_name = "runX"

    # wandb for experiment monitoring
    os.environ['WANDB_NOTES'] = 'test'
    if opt.use_wandb:
        import wandb
        if not use_cuda:
            # os.environ['WANDB_MODE'] = 'dryrun'  # ignore when debugging on cpu
            os.environ['WANDB_TAGS'] = 'CPU'
            wandb.init(anonymous='allow',
                       project="rotated-object-detection", config=opt)
        else:
            wandb.init(anonymous='allow',
                       project="rotated-object-detection", config=opt)

        opt.logger = wandb
        # opt.logger.run.save()
        opt.run_name = opt.logger.run.name  # handle name change in wandb
        atexit.register(_sync_before_exit, opt, wandb)

    return opt


def _sync_before_exit(opt, wandb):
    print("[INFO]: Sync wandb before terminating")
    opt.logger = None  # wandb cant have objects in its config
    wandb.config.update(opt)


def _get_argparser():

    parser = ArgumentParser()
    # training specific
    parser.add_argument('--epochs', default=150, type=int,
                        help='number of epochs to train')
    parser.add_argument('--batch_size', default=256, type=int,
                        help='number of samples per step, have more than one for batch norm')
    parser.add_argument('--learning_rate', default=1e-3, type=float,
                        help='learning rate for all optimizers')
    parser.add_argument('--resume_run', default="None", type=str,
                        help='auto load ckpt')
    # data
    parser.add_argument('--train_len', default=32000, type=int,
                        help='number of samples for training')
    parser.add_argument('--val_len', default=8000, type=int,
                        help='number of samples for validation')
    parser.add_argument('--test_len', default=2, type=int,
                        help='number of samples for testing')
    # output
    parser.add_argument('--use_wandb', default=False, type=bool,
                        help='use wandb to monitor training')
    parser.add_argument('--save_dir', default=f'{os.path.dirname(os.path.abspath(__file__))}/checkpoints', type=str,
                        help='path to save checkpoints')
    # device
    parser.add_argument('--cuda', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='enable cuda if available')
    parser.add_argument('--pin_memory', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='pin memory to device')
    parser.add_argument('--seed', default=400, type=int,
                        help='random seed')
    return parser


if __name__ == "__main__":
    main()
