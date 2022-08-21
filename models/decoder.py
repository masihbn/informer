import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu", device=None):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention

        self.device = device

        self.num_layers = 32
        self.hidden_units = 25
        self.lstm = nn.LSTM(
            input_size=25,
            hidden_size=self.hidden_units,
            batch_first=True,
            num_layers=self.num_layers
        )

        # self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        # self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        y = x = self.norm2(x)

        batch_size = y.shape[0]
        hn = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        cn = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()

        yLSTM, (hn, cn) = self.lstm(y.transpose(-1, 1), (hn, cn))
        cn = cn.float().to(self.device)
        hn = hn.float().to(self.device)
        yLSTM = yLSTM.float().to(self.device)

        # y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        # y = self.dropout(self.conv2(y).transpose(-1,1))
        final_y = self.conv3(yLSTM).transpose(-1, 1)

        return self.norm3(x + final_y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x
