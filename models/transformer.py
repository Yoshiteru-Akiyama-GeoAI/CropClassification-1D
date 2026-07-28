import torch
import torch.nn as nn
import math

##############################################################################
# Transformer
# - Self-Attention
# - Last / Attention / Masked GAP
##############################################################################

class CropTransformer(nn.Module):

    def __init__(
        self,
        n_features=9,
        n_classes=16,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.3,
        batch_first=True,
        norm_first=True,
        att_mode=0
    ):

        super().__init__()

        self.input_proj = nn.Linear(n_features, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.att_mode = att_mode
        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.fc = nn.Linear(d_model, n_classes)

    def forward(
        self,
        x,
        lengths
    ):

        # x (batch, seq_len, 9)
        x = self.input_proj(x)
        batch_size, seq_len, _ = x.shape

        mask = (
            torch.arange(
                seq_len,
                device=x.device
            )[None, :]
            >= lengths[:, None]
        )

        out = self.transformer(x, src_key_padding_mask=mask)

        if self.att_mode == 1:
            # <Attention pooling>
            scores = self.attention_pool(out)   # (B,T,1)
            scores = scores.masked_fill(mask.unsqueeze(-1), -1e9)

            weights = torch.softmax(scores, dim=1)
            context = (out * weights).sum(dim=1)
        elif self.att_mode == 2:
            # <Masked GAP(Global Average Pooling) Based> Average all periods
            valid_mask = (~mask).float()

            context = (out * valid_mask.unsqueeze(-1)).sum(dim=1)
            context /= valid_mask.sum(dim=1, keepdim=True)
        else:
            # <Last Aggregation> only latest period
            idx = lengths - 1
            context = out[torch.arange(out.size(0)), idx]

        logits = self.fc(context)

        return logits