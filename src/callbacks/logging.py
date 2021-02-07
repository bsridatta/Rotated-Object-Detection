from src.callbacks.base import Callback
import torch


class Logging(Callback):
    """Logging and printing metrics"""

    def setup(self, opt, model, **kwargs):
        print(
            f'[INFO]: Start training procedure using device: {opt.device}')
        # log gradients and parameters of the model during training
        if opt.use_wandb:
            opt.logger.watch(model, log='all')

    def on_train_batch_end(self, opt, batch_idx, batch, dataloader, output, l_ship, l_bbox, **kwargs):
        batch_len = len(batch['input'])
        dataset_len = len(dataloader.dataset)
        n_batches = len(dataloader)

        # print to console
        print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}\tL_ship: {:.4f}\tL_bbox: {:.4f}'.format(
            opt.epoch, batch_idx * batch_len,
            dataset_len, 100. * batch_idx / n_batches,
            output, l_ship, l_bbox),
            end='\n')

        # log to wandb
        if opt.use_wandb:
            opt.logger.log({"train_loss": output,
                            "l_ship": l_ship,
                            "l_bbox": l_bbox})

    def on_validation_end(self, opt, output, metrics, l_ship, l_bbox, **kwargs):
        # print and log metrics and loss after validation epoch
        print("Valiation - Loss: {:.4f}\tL_ship: {:.4f}\tL_bbox: {:.4f}".format(
            output, l_ship, l_bbox), end="\t")

        for k in metrics.keys():
            print(f"{k}: {metrics[k]}", end="\t")
            if opt.use_wandb:
                opt.logger.log(metrics, commit=False)
                opt.logger.log({"val_loss": output,
                                "epoch": opt.epoch,
                                "val_l_ship": l_ship,
                                "val_l_bbox": l_bbox})
        print("")

    def on_epoch_end(self, opt, optimizer, **kwargs):
        lr = optimizer.param_groups[0]['lr']
        if opt.use_wandb:
            opt.logger.log({f"LR": lr})
        print("lr @ ", lr)
