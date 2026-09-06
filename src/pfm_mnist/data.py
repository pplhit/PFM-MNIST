from __future__ import annotations

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def make_mnist_loader(
    data_root: str,
    size: int,
    batch_size: int,
    train: bool = True,
    num_workers: int = 2,
    shuffle: bool = True,
) -> DataLoader:
    transform = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
    dataset = datasets.MNIST(root=data_root, train=train, download=True, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train,
    )
