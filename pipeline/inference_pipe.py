import os
import numpy as np
import pandas as pd

# PyTorch
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

# Local Functions
from models import create_model
from dataset import CachedTimeSen2CropDataset
from core.inference import Inference
from core.utils import collate_fn, make_file_list
from core.fileIF import SaveResults


def Pipe_Inference(save_dir, cache_dir, add_doy, model_type, 
                    hidden_dim, att_mode, prm_batch):
    print("Inference started.")

    _, _, _, _, _, _, train_Tm = (
        make_file_list(cache_dir, "train")
    )
    test_file, mean_tt, std_tt, class_to_idx, n_classes, bands, test_Tm = (
        make_file_list(cache_dir, "test")
    )
    test_ds = CachedTimeSen2CropDataset(ds_file=test_file, 
                                        mean=mean_tt, std=std_tt, 
                                        add_doy=add_doy if model_type > 0 else False,)
    print("test_ds size : ", len(test_ds))

    print("Load Test Data...")
    test_loader = DataLoader(
                            test_ds,
                            batch_size=prm_batch,
                            shuffle=False,
                            num_workers=0,
                            pin_memory=True,
                            collate_fn=collate_fn
                            )
    print("test_loader size : ", len(test_loader))
    
    # Preperation
    n_features = test_ds[0][0].shape[-1]   # len(bands) + DOY
    print("n_features : ", n_features)

    # create model and device
    device, model = create_model(model_type=model_type, 
                                n_features=n_features, 
                                n_classes=n_classes, 
                                hidden_dim=hidden_dim, 
                                att_mode=att_mode)
    criterion = nn.CrossEntropyLoss()
    
    print("Inference...")
    result = Inference(
        model=model,
        test_loader=test_loader,
        save_dir=save_dir,
        device=device,
        n_classes=n_classes,
        criterion=criterion
    )
    gts = result["gts"]
    preds = result["eval_preds"]
    
    class_names = list(class_to_idx.keys())
    SaveResults(class_names, gts, preds, save_dir, header="test")
    print("Inference has been finished.\n")
