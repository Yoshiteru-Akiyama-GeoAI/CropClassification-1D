import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import rasterio

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

import yaml

def read_yaml(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        config = yaml.load(file, Loader=yaml.SafeLoader)
        print(config)
    return config

def write_yaml(filepath, config):
    with open(filepath,'w')as f:
        yaml.dump(config, f, encoding='utf-8', default_flow_style=False, allow_unicode=True)

def find_band_file(date_dir, band_name):
    files = list(date_dir.glob(f"*_{band_name}_*.tif"))
    if len(files) == 0:
        raise FileNotFoundError(
            f"Band file not found : {band_name} ({date_dir})"
        )
    return files[0]

def find_label_file(date_dir):
    files = list(date_dir.glob("*_LC_*.tif"))
    if len(files) == 0:
        raise FileNotFoundError(
            f"Cloud file not found : {date_dir}"
        )
    return files[0]

def find_cloud_file(date_dir):
    files = list(date_dir.glob("*_CLD_*.tif"))
    if len(files) == 0:
        raise FileNotFoundError(
            f"Cloud file not found : {date_dir}"
        )
    return files[0]

def save_confusion_matrix(gts, preds, class_names, save_path):
    cm = confusion_matrix(gts, preds)
    print(confusion_matrix(gts, preds))

    fig, ax = plt.subplots(figsize=(12,12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    disp.plot(
        ax=ax,
        xticks_rotation=90,
        cmap="Blues",
        colorbar=False
    )

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_classification_report(gts, preds, class_names, save_path):
    report = classification_report(
        gts,
        preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    pd.DataFrame(report).transpose().to_csv(save_path)

def save_learning_curve(train_losses, val_losses, save_path):
    plt.figure(figsize=(8, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curve")

    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)

    plt.close()

def SaveResults(class_names, gts, preds, save_dir, header,
                train_losses=None, val_losses=None, val_accies=None):

    file_learning_curve = f"{header}_learning_curve.png"
    file_history = f"{header}_history.csv"
    file_confusion_matrix = f"{header}_confusion_matrix.png"
    file_classification_report = f"{header}_classification_report.csv"

    if (
        train_losses is not None
        and val_losses is not None
        and val_accies is not None
    ):
        # Save Learning Curve
        save_learning_curve(
            train_losses,
            val_losses,
            os.path.join(save_dir, file_learning_curve)
        )
        # History
        history = pd.DataFrame({
            "epoch": range(len(train_losses)),
            "train_loss": train_losses,
            "val_loss": val_losses,
            "val_acc": val_accies
        })
        history.to_csv(os.path.join(save_dir, file_history), index=False)
    
    save_confusion_matrix(
        gts,
        preds,
        class_names,
        os.path.join(save_dir, file_confusion_matrix)
    )

    save_classification_report(
        gts,
        preds,
        class_names,
        os.path.join(save_dir, file_classification_report)
    )

    print("ACC", accuracy_score(gts, preds))
    print("Macro F1", f1_score(gts, preds, average="macro", zero_division=0))
    print("Weighted F1", f1_score(gts, preds, average="weighted", zero_division=0))
