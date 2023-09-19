import random

import torch
import numpy as np
from omegaconf import DictConfig
from datasets import load_dataset
from torch.utils.data import DataLoader


def normalize_batch(batch):
    # Assuming batch is a PyTorch tensor with shape [batch, channels, sequence]
    min_val = batch.min()
    max_val = batch.max()
    # Min-Max normalization to [-1, 1]
    batch = 2 * ((batch - min_val) / (max_val - min_val)) - 1
    return batch


def collate_fn(batch):
    # Extract signals
    signals = [item["signal"] for item in batch]
    signals = torch.stack(signals, dim=0)
    # Normalize signals
    signals = normalize_batch(signals)
    return {"signal": signals}


def set_seed(seed):
    # https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloader(cfg: DictConfig, seed: int = None) -> (DataLoader, DataLoader, DataLoader):
    data = load_dataset("roszcz/ecg-segmentation-ltafdb")
    train = data["train"]
    validation = data["validation"]
    test = data["test"]

    columns = ["signal"]

    train.set_format(type="torch", columns=columns)
    validation.set_format(type="torch", columns=columns)
    test.set_format(type="torch", columns=columns)

    generator = torch.Generator().manual_seed(seed) if seed is not None else None

    train_loader = DataLoader(
        train,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        generator=generator,
        worker_init_fn=seed_worker,
        collate_fn=collate_fn,
    )
    validation_loader = DataLoader(
        validation,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        generator=generator,
        worker_init_fn=seed_worker,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        generator=generator,
        worker_init_fn=seed_worker,
        collate_fn=collate_fn,
    )

    return train_loader, validation_loader, test_loader
