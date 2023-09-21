import random

import torch
import numpy as np
from omegaconf import DictConfig
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader


class CustomECGDataset(Dataset):
    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset

    def __len__(self):
        return len(self.hf_dataset) * 4  # Each original sample will produce 4 new samples

    def __getitem__(self, index):
        original_index = index // 4  # Determine the index in the original dataset
        slice_index = index % 4  # Determine which slice (0, 1, 2, or 3)

        sample = self.hf_dataset[original_index]
        start = slice_index * 250
        end = start + 250

        channel1 = torch.tensor(sample["signal"][0][start:end])
        channel2 = torch.tensor(sample["signal"][1][start:end])

        return channel1, channel2


class CombinedECGDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

        # Calculate the cumulative sizes of datasets for indexing purposes
        self.cumulative_sizes = np.cumsum([len(ds) for ds in self.datasets])

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        # Determine which dataset the idx belongs to
        dataset_idx = next(i for i, cum_size in enumerate(self.cumulative_sizes) if idx < cum_size)

        if dataset_idx > 0:
            # Adjust idx to the local index of the dataset it belongs to
            idx -= self.cumulative_sizes[dataset_idx - 1]

        # Convert the idx to a Python int type
        idx = int(idx)

        return self.datasets[dataset_idx][idx]


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


def create_dataloader(
    cfg: DictConfig, seed: int = None, datasets=["train", "validation", "test"]
) -> (DataLoader, DataLoader, DataLoader):
    data = load_dataset("roszcz/ecg-segmentation-ltafdb")

    # Combine the three datasets
    datasets_list = [CustomECGDataset(data[dataset_name]) for dataset_name in datasets]
    combined_dataset = CombinedECGDataset(*datasets_list)

    generator = torch.Generator().manual_seed(seed) if seed is not None else None

    # Create a single DataLoader from the combined dataset
    combined_loader = DataLoader(
        combined_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        generator=generator,
        worker_init_fn=seed_worker,
    )

    return combined_loader
