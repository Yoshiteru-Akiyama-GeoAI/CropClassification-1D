import torch
import torch.nn as nn
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence
)

##############################################################################
# LSTM
# - bidirectional=True
# - Last / Attention / Masked GAP
##############################################################################
class Attention_LSTM(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out, mask=None):
        # lstm_out: [B, T, H]

        score = self.attn(lstm_out).squeeze(-1)  # [B, T]
        
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)

        weight = torch.softmax(score, dim=1)     # [B, T]
        context = torch.bmm(
            weight.unsqueeze(1),                 # [B, 1, T]
            lstm_out                            # [B, T, H]
        ).squeeze(1)                            # [B, H]

        return context, weight

class CropLSTM(nn.Module):

    def __init__(
        self,
        n_features=9,
        n_classes=16,
        hidden_dim=256,
        bidirectional=True,
        att_mode=0
    ):

        super().__init__()

        # dropout : to reduce overfitting
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=bidirectional
        )

        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)

        # Attention : considering time series
        self.att_mode = att_mode
        self.attention = Attention_LSTM(lstm_out_dim)

        self.fc = nn.Linear(lstm_out_dim, n_classes)

    def forward(self, x, lengths):

        packed = pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        packed_out, (h_n, c_n) = self.lstm(packed)
        
        if self.att_mode == 1:
            # <Attention>
            out, _ = pad_packed_sequence(packed_out, batch_first=True)  # [B, T, H]

            # mask（remove padding）
            max_len = out.size(1)
            mask = (
                torch.arange(max_len, device=x.device)[None, :]
                < lengths[:, None]
            )

            context, attn_weights = self.attention(out, mask)
        elif self.att_mode == 2:
            # <Masked GAP>
            out, _ = pad_packed_sequence(packed_out, batch_first=True)

            max_len = out.size(1)
            mask = (
                torch.arange(max_len, device=x.device)[None, :]
                < lengths[:, None]
            ).float()

            context = (out * mask.unsqueeze(-1)).sum(dim=1)
            context /= mask.sum(dim=1, keepdim=True)
        else:
            # <Last aggregation>
            # Last Hidden
            if self.lstm.bidirectional:
                context = torch.cat((h_n[-2], h_n[-1]), dim=1)
            else:
                context = h_n[-1]

        logits = self.fc(context)

        return logits
