import os

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset


def normalize_values(segments, global_max, global_min):
    """Normalize a list of segments (list of lists) to the [0, 1] range using numpy."""
    # Convert segments to a numpy array
    segments_array = np.array(segments)

    # Normalize the entire array
    normalized_array = (segments_array - global_min) / (global_max - global_min)

    # Convert the normalized numpy array back to a list of lists
    normalized_segments = normalized_array.tolist()

    return normalized_segments


def create_dict_from_split(split, rolling_window_size=1024, hop_size=256):
    # Initialize lists to hold data for each window from all tracks
    names = []
    starts = []
    durations = []
    pitches = []
    velocities = []

    for track in split:
        name = f"{track['composer']} {track['title']}"
        total_length = len(track["notes"]["start"])
        # convert starts to dstarts (difference between starts)
        track["notes"]["start"] = np.diff(track["notes"]["start"])
        # Add a zero to the start of the dstarts
        track["notes"]["start"] = np.insert(track["notes"]["start"], 0, 0)

        window_number = 0
        for idx in range(0, total_length - rolling_window_size + 1, hop_size):
            names.append(f"{window_number} {name}")

            starts.append(track["notes"]["start"][idx : idx + rolling_window_size])

            durations.append(track["notes"]["duration"][idx : idx + rolling_window_size])

            pitches.append(track["notes"]["pitch"][idx : idx + rolling_window_size])
            velocities.append(track["notes"]["velocity"][idx : idx + rolling_window_size])
            # change name for next window
            window_number += 1

    # Create a dictionary containing all the data
    data_dict = {
        "name": names,
        "start": starts,
        "duration": durations,
        "pitch": pitches,
        "velocity": velocities,
    }

    return data_dict


if __name__ == "__main__":
    data = load_dataset("roszcz/maestro-v1-sustain")
    rolling_window_size = 60
    hop_size = 15

    train = data["train"]
    validation = data["validation"]
    test = data["test"]

    train_dict = create_dict_from_split(train, rolling_window_size, hop_size)
    validation_dict = create_dict_from_split(validation, rolling_window_size, hop_size)
    test_dict = create_dict_from_split(test, rolling_window_size, hop_size)
    # find global max and min
    global_start_max = max(
        max(np.max(segment) for segment in train_dict["start"]),
        max(np.max(segment) for segment in validation_dict["start"]),
        max(np.max(segment) for segment in test_dict["start"]),
    )
    global_start_min = min(
        min(np.min(segment) for segment in train_dict["start"]),
        min(np.min(segment) for segment in validation_dict["start"]),
        min(np.min(segment) for segment in test_dict["start"]),
    )

    global_duration_max = max(
        max(np.max(segment) for segment in train_dict["duration"]),
        max(np.max(segment) for segment in validation_dict["duration"]),
        max(np.max(segment) for segment in test_dict["duration"]),
    )
    global_duration_min = min(
        min(np.min(segment) for segment in train_dict["duration"]),
        min(np.min(segment) for segment in validation_dict["duration"]),
        min(np.min(segment) for segment in test_dict["duration"]),
    )
    print("start max, start min")
    print(global_start_max, global_start_min)
    print("duration max, duration min")
    print(global_duration_max, global_duration_min)

    train_dict["start"] = normalize_values(train_dict["start"], global_start_max, global_start_min)
    train_dict["duration"] = normalize_values(train_dict["duration"], global_duration_max, global_duration_min)

    validation_dict["start"] = normalize_values(validation_dict["start"], global_start_max, global_start_min)
    validation_dict["duration"] = normalize_values(validation_dict["duration"], global_duration_max, global_duration_min)

    test_dict["start"] = normalize_values(test_dict["start"], global_start_max, global_start_min)
    test_dict["duration"] = normalize_values(test_dict["duration"], global_duration_max, global_duration_min)

    DATASET_NAME = "SneakyInsect/maestro-rollingsplit"
    HUGGINGFACE_TOKEN = os.environ["HUGGINGFACE_TOKEN"]

    train_dataset = Dataset.from_dict(train_dict)
    validation_dataset = Dataset.from_dict(validation_dict)
    test_dataset = Dataset.from_dict(test_dict)

    dataset_dict = DatasetDict({"train": train_dataset, "validation": validation_dataset, "test": test_dataset})

    dataset_dict.push_to_hub(DATASET_NAME, token=HUGGINGFACE_TOKEN)
