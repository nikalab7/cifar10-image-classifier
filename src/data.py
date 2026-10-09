"""CIFAR-10 loading with a leakage-safe train / validation / test protocol.

* The official 10,000-image test set is used exactly once, after model selection.
* 5,000 of the 50,000 training images are held out as a validation set. They come from
  the same files as the training images but are always read WITHOUT augmentation.
"""
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")

# Per-channel mean / std of the CIFAR-10 training images.
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def build_transforms(augment: bool, image_size: int = 32):
    """Return (train_transform, eval_transform)."""
    resize = [transforms.Resize(image_size)] if image_size != 32 else []
    normalize = [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    eval_tf = transforms.Compose(resize + normalize)
    if not augment:
        return eval_tf, eval_tf
    train_tf = transforms.Compose(
        resize
        + [transforms.RandomCrop(image_size, padding=image_size // 8),
           transforms.RandomHorizontalFlip()]
        + normalize
    )
    return train_tf, eval_tf


def get_loaders(data_dir="data", augment=True, image_size=32, batch_size=128,
                val_size=5000, seed=0, num_workers=2, fake=False):
    """Build train / validation / test DataLoaders.

    fake=True swaps CIFAR-10 for a tiny random dataset. It exists only to check
    that the code runs end to end without downloading anything.
    """
    train_tf, eval_tf = build_transforms(augment, image_size)

    if fake:
        full_train = datasets.FakeData(600, (3, 32, 32), 10, transform=train_tf)
        full_val = datasets.FakeData(600, (3, 32, 32), 10, transform=eval_tf)
        test_set = datasets.FakeData(200, (3, 32, 32), 10, transform=eval_tf,
                                     random_offset=10_000)
        val_size = 100
    else:
        full_train = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_tf)
        # Same images again, but with the evaluation transform (no augmentation).
        full_val = datasets.CIFAR10(data_dir, train=True, download=False, transform=eval_tf)
        test_set = datasets.CIFAR10(data_dir, train=False, download=True, transform=eval_tf)

    # Fixed, seeded split of the training images into train / validation indices.
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(full_train), generator=generator).tolist()
    val_idx, train_idx = permutation[:val_size], permutation[val_size:]

    kwargs = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(Subset(full_train, train_idx), batch_size=batch_size,
                              shuffle=True, **kwargs)
    val_loader = DataLoader(Subset(full_val, val_idx), batch_size=256, shuffle=False, **kwargs)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, **kwargs)
    return train_loader, val_loader, test_loader
