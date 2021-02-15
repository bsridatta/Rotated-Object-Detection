from torch.utils.data import Dataset
from src.rotated_ship_data import make_data
import torch
import numpy as np
from tqdm import tqdm


class Ships(Dataset):
    """ship datasets with has ship labels
    Keyword Arguments:
        n_samples {int} -- items in dataset, here, items per epoch (default: {1000})
        pre_load {bool} -- to make all items at once and query for each step (default: {False})

    Returns:
        sample {Tenosr} -- p_ship, x, y, yaw, h, w
    """

    def __init__(self, n_samples=1000, pre_load=False):
        self.n_samples = n_samples
        self.pre_load = pre_load
        if pre_load:
            images, labels = make_batch(n_samples)
            # row, col -> n_channel,row,col
            inp = torch.tensor(images, dtype=torch.float32)
            self.inps = inp[:, None, :, :]

            # x,y,yaw,h,w -> p(ship),x,y,yaw,h,w
            target = torch.tensor(labels, dtype=torch.float32)
            has_ship = (~torch.isnan(target[:, 0])).float().reshape(-1, 1)
            self.targets = torch.cat((has_ship, target), dim=1)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if self.pre_load:
            inp = self.inps[idx]
            target = self.targets[idx]
        else:
            image, label = make_data()

            # row, col -> n_channel,row,col
            inp = torch.tensor(image, dtype=torch.float32)
            inp = inp[None, :, :]

            # x,y,yaw,h,w -> p(ship),x,y,yaw,h,w
            target = torch.tensor(label, dtype=torch.float32)
            has_ship = (~torch.isnan(target[0])).float().reshape(1)
            target = torch.cat((has_ship, target), dim=0)

        sample = {"input": inp, "target": target}

        return sample

# Used for simple experiment


def make_batch(batch_size):
    """Used only when pre_load = True

    Arguments:
        batch_size {int} -- number of samples to generate

    Returns:
        images, labels -- images with/without ship, label with has_ship
    """
    imgs, labels = zip(*[make_data() for _ in tqdm(range(batch_size))])
    imgs = np.stack(imgs)
    labels = np.stack(labels)
    return imgs, labels
