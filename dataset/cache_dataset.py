import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import Dataset


class CachedTimeSen2CropDataset(Dataset):

    def __init__(
        self,
        ds_file,
        mean,
        std,
        add_doy=True
    ):
        self.ds_file = ds_file
        self.data = torch.load(
            self.ds_file,
            map_location="cpu",
            weights_only=True
        )
        self.has_label = "y" in self.data[0]
        self.mean = mean.clone().detach()
        self.std = std.clone().detach()
        self.add_doy = add_doy

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        x = sample["x"]          # (T, 9)
        y = sample["y"]
        length = torch.tensor(sample["length"], dtype=torch.long)
        
        # ★ Normalization
        x = (x - self.mean) / (self.std + 1e-8)

        # ★ add DOY (NOT Seasonal Ferquency)
        if self.add_doy == 1:
            T = x.size(0)
            doy = torch.arange(T, dtype=torch.float32) + 1
            # single
            doy = doy / 365.0
            doy = doy.unsqueeze(-1)   # (T, 1)
            # concatenate
            x = torch.cat([x, doy], dim=-1)
        elif self.add_doy == 2:
            T = x.size(0)
            doy = torch.arange(T, dtype=torch.float32) + 1
            # sin/cos
            angle = 2 * torch.pi * doy / 365.0
            doy_sin = torch.sin(angle)
            doy_cos = torch.cos(angle)
            doy = torch.stack([doy_sin, doy_cos], dim=-1)  # (T, 2)
            # concatenate
            x = torch.cat([x, doy], dim=-1)

        # dummy
        px = torch.tensor(-1)
        py = torch.tensor(-1)

        return x, y, px, py, length
