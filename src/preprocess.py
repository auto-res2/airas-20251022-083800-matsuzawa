from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

CACHE_DIR = Path(".cache").absolute()


def get_dataloaders(cfg) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return train, val, test loaders according to cfg."""
    root = CACHE_DIR / cfg.dataset.name
    root.mkdir(parents=True, exist_ok=True)

    tfs = [transforms.ToTensor()]
    if getattr(cfg.dataset, "normalization", True):
        # Fashion-MNIST normalisation constants
        tfs.append(transforms.Normalize((0.5,), (0.5,)))
    transform = transforms.Compose(tfs)

    full_train = datasets.FashionMNIST(root=str(root), train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST(root=str(root), train=False, download=True, transform=transform)

    train_size = cfg.dataset.train_size
    val_size = cfg.dataset.val_size

    train_subset, val_subset = random_split(full_train, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    num_workers = 4 if torch.cuda.is_available() else 0
    pin_mem = torch.cuda.is_available()

    train_loader = DataLoader(train_subset, batch_size=cfg.training.batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_mem)
    val_loader = DataLoader(val_subset, batch_size=cfg.training.batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_mem)
    test_loader = DataLoader(test_set, batch_size=cfg.training.batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_mem)
    return train_loader, val_loader, test_loader