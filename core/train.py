import os
import numpy as np
import time

from tqdm import tqdm

# PyTorch
import torch
import torch.nn as nn

from models import save_model

class EarlyStopping:

    def __init__(
        self,
        patience=3,
        mode="min",
        delta=0.0,
        path="best_model.pt"
    ):
        """
        mode:
            "min" -> val_loss
            "max" -> val_acc, macro_f1
        """

        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.path = path

        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(
        self,
        metric,
        model
    ):

        if self.mode == "min":
            score = -metric
        else:
            score = metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(
                f"EarlyStopping counter: "
                f"{self.counter}/{self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)

    def save_checkpoint(self, model):
        torch.save(
            model.state_dict(),
            self.path
        )

        print(
            "Validation improved. "
            "Saving model."
        )

def Training(
    model,
    train_loader,
    valid_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs,
    es_patience=3,
    save_dir="./checkpoints"
):

    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    val_accies = []
    best_gts = None
    best_preds = None

    early_stopping = EarlyStopping(
                        patience=es_patience,
                        mode="min",
                        path=os.path.join(save_dir, "es_model.pth")
                    )

    for epoch in range(num_epochs):
        # =========================
        # TRAIN
        # =========================
        model.train()
        train_loss = 0.0

        for x, y, _, _, lengths in tqdm(train_loader, desc=f"Train {epoch+1}", mininterval=0.5):
            x = x.to(device)
            y = y.to(device)
            lengths = lengths.to(device)

            optimizer.zero_grad()

            pred = model(x, lengths)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # =========================
        # VALIDATION
        # =========================
        model.eval()
        val_loss = 0.0

        preds = []
        gts = []

        with torch.no_grad():
            for x, y, _, _, lengths in valid_loader:

                x = x.to(device)
                y = y.to(device)
                lengths = lengths.to(device)

                pred = model(x, lengths)
                loss = criterion(pred, y)
                val_loss += loss.item()

                preds.extend(pred.argmax(1).cpu().numpy())
                gts.extend(y.cpu().numpy())

        val_loss /= len(valid_loader)
        val_losses.append(val_loss)
        val_acc = (np.array(preds) == np.array(gts)).mean()
        val_accies.append(val_acc)
        scheduler.step(val_loss)

        # =========================
        # LOG
        # =========================
        print(f"[Epoch {epoch+1}] "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} ")

        # SAVE in every epoch
        save_model(model, optimizer, epoch, save_dir, "latest_model.pth")

        # =========================
        # BEST MODEL UPDATE based on Val_loss
        # =========================
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, optimizer, epoch, save_dir, "best_model.pth")
            best_gts = gts.copy()
            best_preds = preds.copy()
            print(f"  → Best model updated (val_loss={val_loss:.4f})")
        
        # Early Stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    return gts, preds, best_gts, best_preds, train_losses, val_losses, val_accies

