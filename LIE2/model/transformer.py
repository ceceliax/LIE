
import torch.nn as nn
import torch
from torch.autograd import Variable
import math
import numpy as np

class PositionalEncoding(nn.Module):

    # def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
    def __init__(self, d_model, dropout, max_len):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x) :
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class transformer(nn.Module):

    def __init__(self,  output_size, embed_dim, nhead, num_layer, sequence_length):
        super(transformer, self).__init__()
        self.d_model = embed_dim
        self.nhead = nhead
        self.sequence_length = sequence_length
        self.num_layer = num_layer
        self.output_size = output_size
        self.pos_encoder = PositionalEncoding(d_model=self.d_model, dropout=0.5, max_len=self.sequence_length)
        self.encoder_layer = nn.TransformerEncoderLayer(self.d_model, self.nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, self.num_layer)

        # self.label = nn.Linear(hidden_size * self.layer_size * self.sequence_length, 9)#128
        self.label = nn.Linear(embed_dim, self.output_size)


    def forward(self, x):
        batch_size = x.size(0)  ### 16
        x = torch.movedim(x, 0, 1)
        pos_en = self.pos_encoder(x)
        pos_en = torch.movedim(pos_en, 0, 1)

        out = self.encoder(pos_en)
        output = torch.sum(out, 1)
        output = output.reshape(batch_size, -1)
        logits = self.label(output)
        return logits