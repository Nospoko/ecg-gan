import os

import numpy as np
from datasets import DatasetDict, load_dataset


def filter_func(example):
    # count number of 1s in the mask
    mask = example["mask"]
    count = np.sum(mask)
    # filter out examples with less than 10 1s
    return count >= 10


if __name__ == "__main__":
    dataset = load_dataset("roszcz/ecg-segmentation-ltafdb")

    filtered_dataset = {split: dataset[split].filter(filter_func) for split in dataset.keys()}
    print("filtered out useless examples")
    filtered_dataset = {split: filtered_dataset[split].remove_columns(["mask"]) for split in filtered_dataset.keys()}
    print("removed mask column")

    filtered_dataset = DatasetDict(filtered_dataset)

    token = os.environ["HUGGINGFACE_TOKEN"]
    filtered_dataset.push_to_hub("SneakyInsect/ltafdb_preprocessed", token=token)
