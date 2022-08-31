import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderLayerGRU(nn.Module):
    def __init__(self, pred_len, input_size, hidden_size,
                 num_layers, seq_length, device=None):
        super(DecoderLayerGRU, self).__init__()

        self.pred_len = pred_len
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x, cross=None, x_mask=None, cross_mask=None):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        # Propagate input through LSTM
        _, h_out = self.gru(x.transpose(-1, 1), h_0)
        h_out = h_out.view(-1, self.hidden_size)
        out = self.fc(h_out)

        return out


class DecoderLayerLSTM(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model,
                 hidden_units, input_size, num_layers,
                 dropout=0.1, activation="relu", device=None):
        super(DecoderLayerLSTM, self).__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention

        self.device = device

        self.input_size = input_size
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_units,
            batch_first=True,
            num_layers=self.num_layers
        )

        self.conv = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1)
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
        hn = torch.zeros(self.num_layers, batch_size, self.hidden_units, device=self.device).requires_grad_()
        cn = torch.zeros(self.num_layers, batch_size, self.hidden_units, device=self.device).requires_grad_()

        y, (hn, cn) = self.lstm(y.transpose(-1, 1), (hn, cn))
        y = self.conv(y).transpose(-1, 1)

        return self.norm3(x + y)


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu", device=None):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
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
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        return self.norm3(x+y)


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
