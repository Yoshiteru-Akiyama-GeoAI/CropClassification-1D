import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from models import load_model

def Inference(
    model,
    test_loader,
    save_dir,
    device,
    n_classes,
    criterion=None
):
    # =========================
    # Load model
    # =========================
    model = load_model(model, save_dir, "best_model.pth", device)

    preds = []
    probs = []
    pxs = []
    pys = []
    eval_preds = []
    gts = []
    test_loss = 0.0

    # =========================
    # Inference
    # =========================
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference", mininterval=1.0):
            x, y, px, py, lengths = batch
            x = x.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
            output = model(x, lengths)

            valid = (y >= 0) & (y < n_classes)
            if valid.any():
                if criterion is not None:
                    loss = criterion(output[valid], y[valid])
                    test_loss += loss.item()
                gts.extend(y[valid].cpu().numpy())

            # save coordinates
            pxs.extend(px.cpu().numpy())
            pys.extend(py.cpu().numpy())

            prob = torch.softmax(output, dim=1)
            pred = output.argmax(dim=1)
            preds.extend(pred.cpu().numpy())
            probs.extend(prob.cpu().numpy())
            eval_preds.extend(pred[valid].cpu().numpy())

    preds = np.array(preds)
    probs = np.array(probs)

    # =========================
    # Evaluation
    # =========================
    if len(gts) > 0:
        gts = np.array(gts)
        eval_preds = np.array(eval_preds)
        result = {
            "preds": preds,
            "probs": probs,
            "gts": gts,
            "eval_preds": eval_preds,
            "px": np.array(pxs),
            "py": np.array(pys)
        }

        if criterion is not None:
            result["loss"] = test_loss / len(test_loader)

        result["acc"] = accuracy_score(gts, eval_preds)
        print(f"Test Acc : {result['acc']:.4f}")

        if "loss" in result:
            print(f"Test Loss: {result['loss']:.4f}")
    else:
        # no label
        result = {
            "preds": preds,
            "probs": probs,
            "eval_preds": eval_preds,
            "px": np.array(pxs),
            "py": np.array(pys)
        }
    
    return result
