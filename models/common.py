import torch
import os

from models.lstm import CropLSTM
from models.tcn import CropTCN
from models.transformer import CropTransformer

import warnings
warnings.filterwarnings(
    "ignore",
    message="The PyTorch API of nested tensors is in prototype stage.*"
)

##############################################################################
# Common Functions
##############################################################################
def get_device():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(device)
    return device

def create_model(model_type, n_features, n_classes, hidden_dim, att_mode=0):
    device = get_device()

    match model_type:
        case 0:
            model = CropLSTM(n_features=n_features, n_classes=n_classes, 
                            hidden_dim=hidden_dim, att_mode=att_mode).to(device)
        case 1:
            model = CropTCN(n_features=n_features, n_classes=n_classes, 
                            hidden_dim=hidden_dim, att_mode=att_mode).to(device)
        case 2:
            model = CropTransformer(n_features=n_features, n_classes=n_classes, 
                            d_model=hidden_dim, att_mode=att_mode).to(device)
        case _:
            raise ValueError(f"Unknown model_type: {model_type}")

    return device, model

def save_model(model, optimizer, epoch, save_dir, model_name):
    save_path = os.path.join(save_dir, model_name)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch+1
    }, save_path)

def load_model(model, save_dir, model_name, device):
    load_path = os.path.join(save_dir, model_name),
    checkpoint = torch.load(load_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model
