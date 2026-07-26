import os
import glob
import numpy as np
import pandas as pd
import random
import torch
from pathlib import Path
from tqdm import tqdm

from torch.nn.utils.rnn import (
    pad_sequence
)

def ReadDataset(data_path):
    # Read CSV file
    root = Path(data_path)
    samples = []

    for tile_dir in root.iterdir():
        if not tile_dir.is_dir():
            continue

        for class_dir in tqdm(tile_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name

            for csv_file in class_dir.glob("*.csv"):
                samples.append((csv_file, class_name, tile_dir.name))
    
    print("samples =", len(samples))
    return samples

def MakeLabels(samples):
    classes = sorted(
        list(
            set(
                label
                for _, label, _ in samples
            )
        ),
        key=int
    )

    class_to_idx = {c:i for i,c in enumerate(classes)}
    idx_to_class = {i:cls for cls, i in class_to_idx.items()}

    print("class_to_idx : ", class_to_idx)
    print("idx_to_class : ", idx_to_class)
    return class_to_idx, idx_to_class

def SplitDataset(samples, valid_tile, test_tile):
    train_samples = []
    valid_samples = []
    test_samples = []

    for sample in samples:
        tile = sample[2]

        if tile == valid_tile:
            valid_samples.append(sample)
        elif tile == test_tile:
            test_samples.append(sample)
        else:
            train_samples.append(sample)

    print("train_samples=", len(train_samples))
    print("valid_samples=", len(valid_samples))
    print("test_samples=", len(test_samples))

    return train_samples, valid_samples, test_samples


import random
from sklearn.model_selection import train_test_split

def SplitDataset_RandData(
        samples,
        train_ratio=0.7,
        valid_ratio=0.2,
        test_ratio=0.1,
        seed=42):

    assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6

    labels = [s[1] for s in samples]

    train_samples, temp_samples = train_test_split(
        samples,
        test_size=(1.0-train_ratio),
        random_state=42,
        stratify=labels
    )

    temp_labels = [s[1] for s in temp_samples]

    remain_ratio = valid_ratio + test_ratio
    test_size = test_ratio / remain_ratio
    valid_samples, test_samples = train_test_split(
        temp_samples,
        test_size=test_size,
        random_state=42,
        stratify=temp_labels
    )
    total = len(samples)

    print(f"total={total}")
    print(f"train_samples={len(train_samples)}")
    print(f"valid_samples={len(valid_samples)}")
    print(f"test_samples={len(test_samples)}")

    return (
        train_samples,
        valid_samples,
        test_samples
    )

def SplitDataset_RandMesh(samples, 
                train_ratio=0.7,
                valid_ratio=0.2,
                test_ratio=0.1,
                seed=42):

    assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6

    tiles = sorted(list(set(tile for _, _, tile in samples)))
    num_tiles = len(tiles)
    print("num_tiles={}".format(num_tiles))

    # shuffle
    rng = random.Random(seed)
    rng.shuffle(tiles)

    num_tiles = len(tiles)
    n_train = int(num_tiles * train_ratio)
    n_valid = int(num_tiles * valid_ratio)

    train_tiles = set(tiles[:n_train])
    valid_tiles = set(tiles[n_train:n_train+n_valid])
    test_tiles = set(tiles[n_train+n_valid:])

    train_samples = [s for s in samples if s[2] in train_tiles]
    valid_samples = [s for s in samples if s[2] in valid_tiles]
    test_samples = [s for s in samples if s[2] in test_tiles]

    print("train_samples=", len(train_samples))
    print("valid_samples=", len(valid_samples))
    print("test_samples=", len(test_samples))
    
    return train_samples, valid_samples, test_samples

def collate_fn(batch):
    xs = [b[0] for b in batch]
    ys = torch.stack([b[1] for b in batch])
    pxs = torch.stack([b[2] for b in batch])
    pys = torch.stack([b[3] for b in batch])
    lengths = torch.stack([b[4] for b in batch])
    xs = pad_sequence(xs, batch_first=True)

    return xs, ys, pxs, pys, lengths

def make_file_list(cache_dir, ds_name):
    ds_file = os.path.join(cache_dir, "cache_{}.pt".format(ds_name))

    meta = torch.load(
        os.path.join(cache_dir, "meta_{}.pt".format(ds_name)),
        map_location="cpu",
        weights_only=True
    )

    mean = meta["mean"]
    std = meta["std"]
    class_to_idx = meta["class_to_idx"]
    n_classes = meta["n_classes"]
    bands = meta["bands"]

    if meta.get("max_T") is None:
        max_T = 0
    else:
        max_T = meta["max_T"]

    print("\n===== META READ =====")
    print("Directory : ", cache_dir)
    print("ds_file : ", ds_file)
    print("mean:", mean)
    print("std :", std)
    print("class_to_idx : ", class_to_idx)
    print("n_classes :", n_classes)
    print("bands :", bands)

    return ds_file, mean, std, class_to_idx, n_classes, bands, max_T

