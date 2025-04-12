import torch
from torch import nn
import math
import torch.nn.functional as F


d_model = 512
num_heads = 8
drop_prob = 0.1
batch_size = 30 # for mini-batch gradient descent
max_sequence_length = 200
ffn_hidden = 2048 # generally 512, although more the merrier
num_layers = 5 # no of transformer encoder units in our network

class Encoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers):
        super().__init__()
        self.layers = nn.Sequential(*[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob) for _ in range(num_layers)])

    def forward(self, x):
        x = self.layers(x)
        return x
