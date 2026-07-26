import torch
import torch.nn as nn


##############################################################################
# TCN
# - non-causal
# - Last / Attention / Masked GAP
##############################################################################
import torch.nn.functional as F

class Attention_TCN(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x, mask=None):
        # x: [B, T, H], mask: [B, T]
        scores = self.v(torch.tanh(self.W(x))).squeeze(-1)  # [B, T]

        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)

        attn = F.softmax(scores, dim=1)  # [B, T]
        context = torch.sum(x * attn.unsqueeze(-1), dim=1)  # [B, H]

        return context, attn

class TemporalBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation,
        dropout=0.5
    ):
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x):
        out = self.conv1(x)
        # crop extra padding to keep sequence length
        out = out[:, :, :x.size(2)]

        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = out[:, :, :x.size(2)]

        out = self.relu(out)
        out = self.dropout(out)

        res = x if self.downsample is None else self.downsample(x)

        return self.relu(out + res)

class CropTCN(nn.Module):

    def __init__(
        self,
        n_features=9,
        n_classes=16,
        hidden_dim=64,
        kernel_size=3,
        n_layers=3,
        att_mode=0
    ):

        super().__init__()

        layers = []

        in_ch = n_features

        for i in range(n_layers):
            dilation = 2 ** i

            layers.append(
                TemporalBlock(
                    in_channels=in_ch,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=0.3
                )
            )

            in_ch = hidden_dim

        self.tcn = nn.Sequential(*layers)

        self.att_mode = att_mode
        self.attention = Attention_TCN(hidden_dim)

        self.fc = nn.Linear(hidden_dim, n_classes)

    def forward(self, x, lengths):

        # Conv1D (x: [B,T,F])
        x = x.transpose(1, 2)
        # [B,H,T]
        out = self.tcn(x)
        # [B,T,H]
        out = out.transpose(1, 2)
        # common mask
        max_len = out.size(1)
        mask = (
            torch.arange(max_len, device=out.device)[None, :]
            < lengths[:, None]
        )

        if self.att_mode == 1:
            # <Attention>
            context, attn_weights = self.attention(out, mask)
        elif self.att_mode == 2:
            # <Masked GAP(Global Average Pooling)>
            mask = mask.float()
            context = (out * mask.unsqueeze(-1)).sum(dim=1)
            context = context / mask.sum(dim=1, keepdim=True)
        else:
            # <Last aggregation>
            idx = lengths - 1
            context = out[torch.arange(out.size(0)), idx]

        logits = self.fc(context)

        return logits
