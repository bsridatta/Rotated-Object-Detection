from torch.utils.data import DataLoader
from src.dataset import Ships


def train_dataloader(opt):
    print("[INFO]: Train dataloader called")
    dataset = Ships(n_samples=opt.train_len)
    sampler = None
    shuffle = True
    loader = DataLoader(dataset=dataset,
                        batch_size=opt.batch_size,
                        num_workers=opt.num_workers,
                        pin_memory=opt.pin_memory,
                        sampler=sampler,
                        shuffle=shuffle)
    print("samples - ", len(dataset))
    return loader

def val_dataloader(opt):
    print("[INFO]: Validation dataloader called")
    dataset = Ships(n_samples=opt.val_len)
    sampler = None
    shuffle = True
    loader = DataLoader(dataset=dataset,
                        batch_size=opt.batch_size,
                        num_workers=opt.num_workers,
                        pin_memory=opt.pin_memory,
                        sampler=sampler,
                        shuffle=shuffle)
    print("samples - ", len(dataset))
    return loader

def test_dataloader(opt):
    print("[INFO]: Test dataloader called")
    dataset = Ships(n_samples=opt.test_len)
    sampler = None
    shuffle = True
    loader = DataLoader(dataset=dataset,
                        batch_size=opt.batch_size,
                        num_workers=opt.num_workers,
                        pin_memory=opt.pin_memory,
                        sampler=sampler,
                        shuffle=shuffle)
    print("samples - ", len(dataset))
    return loader 