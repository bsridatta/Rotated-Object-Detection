import os

import torch

from callbacks.base import Callback


class ModelCheckpoint(Callback):
    def __init__(self):
        self.val_loss_min = float("inf")

    def setup(self, opt, model, optimizer, **kwargs):
        # Save model code to wandb
        if opt.use_wandb:
            opt.logger.save(f"{os.path.dirname(os.path.abspath(__file__))}/models/*")

        # Resume training
        if opt.resume_run not in "None":
            state = torch.load(
                f"{opt.save_dir}/{opt.resume_run}.pt", map_location=opt.device
            )
            print(
                f'[INFO] Loaded Checkpoint {opt.resume_run}: @ epoch {state["epoch"]}'
            )
            model.load_state_dict(state["model_state_dict"])

            # Optimizers
            optimizer_state_dic = torch.load(
                f"{opt.save_dir}/{opt.resume_run}_optimizer.pt", map_location=opt.device
            )
            optimizer.load_state_dict(optimizer_state_dic)

    def on_epoch_end(self, opt, val_loss, model, optimizer, epoch, **kwargs):
        # track val loss and save model when it decreases
        if val_loss < self.val_loss_min and opt.device != "cpu":
            self.val_loss_min = val_loss

            try:
                state_dict = model.module.state_dict()
            except AttributeError:
                state_dict = model.state_dict()

            state = {
                "epoch": epoch,
                "val_loss": val_loss,
                "model_state_dict": state_dict,
            }

            # model
            torch.save(state, f"{opt.save_dir}/{opt.run_name}.pt")
            if opt.use_wandb:
                opt.logger.save(f"{opt.save_dir}/{opt.run_name}.pt")
            print(f"[INFO] Saved pt: {opt.save_dir}/{opt.run_name}.pt")

            del state

            # Optimizer
            torch.save(
                optimizer.state_dict(), f"{opt.save_dir}/{opt.run_name}_optimizer.pt"
            )
            if opt.use_wandb:
                opt.logger.save(f"{opt.save_dir}/{opt.run_name}_optimizer.pt")
            print(f"[INFO] Saved pt: {opt.save_dir}/{opt.run_name}_optimizer.pt")
