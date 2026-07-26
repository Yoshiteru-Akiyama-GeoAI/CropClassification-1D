import os
import numpy as np

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight

# Local Functions
from models import create_model
from dataset import CachedTimeSen2CropDataset
from core.train import Training
from core.utils import (
    collate_fn, 
    make_file_list
)
from core.fileIF import SaveResults


def Pipe_Training(save_dir, cache_dir, prm_add_doy,
                            model_type, hidden_dim, 
                            att_mode, prm_class_weights,
                            prm_batch, prm_lr, prm_weight, 
                            prm_epoch, prm_es_patience):
    print("Training and Validation started.")

    # Preparation for training
    train_file, mean_t, std_t, class_to_idx, n_classes_t, bands, _ = (
        make_file_list(cache_dir, "train")
    )
    train_ds = CachedTimeSen2CropDataset(ds_file=train_file, 
                                        mean=mean_t, std=std_t, 
                                        add_doy=prm_add_doy)
    print("train_ds size : ", len(train_ds))

    valid_file, mean_v, std_v, class_to_idx, n_classes_v, bands, _ = (
        make_file_list(cache_dir, "valid")
    )
    valid_ds = CachedTimeSen2CropDataset(ds_file=valid_file, 
                                        mean=mean_v, std=std_v, 
                                        add_doy=prm_add_doy)
    print("valid_ds size : ", len(valid_ds))

    # -----------------------------------
    # check n_classes
    if n_classes_t != n_classes_v:
        raise ValueError(
            f"Class number : Train '{n_classes_t}' / Valid '{n_classes_v}' not matched"
        )
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
        exit()
    else:
        # Data Loader
        train_loader = DataLoader(
            train_ds,
            batch_size=prm_batch,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_fn
        )
        print("train_loader size : ", len(train_loader))
        valid_loader = DataLoader(
            valid_ds,
            batch_size=prm_batch,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_fn
        )
        print("valid_loader size : ", len(valid_loader))
        n_classes = n_classes_t

    # Preperation
    n_features = train_ds[0][0].shape[-1]   # len(bands) + DOYs
    print("n_features : ", n_features)
    # create model and device
    device, model = create_model(model_type=model_type, 
                                n_features=n_features, 
                                n_classes=n_classes, 
                                hidden_dim=hidden_dim, 
                                att_mode=att_mode)

    # criterion
    if prm_class_weights:
        print("Calculate Class Weights.")

        train_labels = []
        for x, y, px, py, length in tqdm(train_ds):
            train_labels.append(y.item())
        train_labels = np.array(train_labels)
        classes, class_counts = np.unique(train_labels, return_counts=True)

        match prm_class_weights:
            case 1:
                # [a] balanced weight
                weights = compute_class_weight(
                    class_weight="balanced",
                    classes=classes,
                    y=train_labels
                )
            case 2:
                # [b] normalized weight
                weights = 1.0 / np.power(class_counts, 0.3)
                weights = weights / weights.mean()

        class_weights = torch.tensor(weights, dtype=torch.float32)
        print("class_weights=", class_weights)
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.05)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=prm_lr, weight_decay=prm_weight)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    gts, preds, best_gts, best_preds, train_losses, val_losses, val_accies = Training(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=prm_epoch,
        es_patience=prm_es_patience,
        save_dir=save_dir
    )
    print("Training and Validation has been finished.\n")
    
    # Save Results
    print("Evaluation started.")
    class_names = list(class_to_idx.keys())
    SaveResults(class_names, gts, preds, save_dir, 
                header="train",
                train_losses=train_losses,
                val_losses=val_losses, 
                val_accies=val_accies)
    SaveResults(class_names, best_gts, best_preds, save_dir, header="train_best")
    
    print("Evaluation has been finished.\n")
