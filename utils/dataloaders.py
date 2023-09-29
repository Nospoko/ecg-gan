import random

import torch
import numpy as np
from omegaconf import DictConfig
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, concatenate_datasets


class CustomECGDataset(Dataset):
    def __init__(self, hf_dataset, data_size):
        self.hf_dataset = hf_dataset
        self.data_multiplier = 1000 // data_size

        # It's computationally expensive to calculate the global min and max
        # The values are taken from data_showcase.ipynb
        self.global_max = [9.494109153747559, 7.599456787109375]
        self.global_min = [-10.515237808227539, -7.820725917816162]

        # check if it's a power of 2
        assert (self.data_multiplier & (self.data_multiplier - 1) == 0) and self.data_multiplier != 0

    def __len__(self):
        return len(self.hf_dataset) * self.data_multiplier  # Each original sample will produce data_multiplier samples

    def __getitem__(self, index):
        original_index = index // self.data_multiplier  # Determine the index in the original dataset
        slice_index = index % self.data_multiplier  # Determine which slice (depending on data_multiplier)

        new_size = 1000 // self.data_multiplier
        sample = self.hf_dataset[original_index]
        start = slice_index * new_size
        end = start + new_size

        # Extract channels
        channel1 = np.array(sample["signal"][0][start:end])
        channel2 = np.array(sample["signal"][1][start:end])

        # Normalize each channel in the range [-1, 1]
        channel1 = 2 * ((channel1 - self.global_min[0]) / (self.global_max[0] - self.global_min[0])) - 1
        channel2 = 2 * ((channel2 - self.global_min[1]) / (self.global_max[1] - self.global_min[1])) - 1

        # Convert to tensor
        channel1 = torch.tensor(channel1, dtype=torch.float32)
        channel2 = torch.tensor(channel2, dtype=torch.float32)

        return channel1, channel2


def set_seed(seed, deterministic=True):
    # https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(deterministic)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloader(
    cfg: DictConfig, seed: int = None, splits=["train", "validation", "test"]
) -> (DataLoader, DataLoader, DataLoader):
    # Use cleaned up dataset ~3% less samples, but less noise
    datasets = [load_dataset("SneakyInsect/ltafdb_preprocessed", split=split) for split in splits]
    dataset = concatenate_datasets(datasets)

    combined_dataset = CustomECGDataset(dataset, cfg.data.size)

    generator = torch.Generator().manual_seed(seed) if seed is not None else None

    # Create a single DataLoader from the combined dataset
    combined_loader = DataLoader(
        combined_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        generator=generator,
        worker_init_fn=seed_worker,
        num_workers=cfg.train.num_workers,
    )

    return combined_loader
