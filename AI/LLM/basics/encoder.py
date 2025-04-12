# Code picked up by following the video:
# https://www.youtube.com/watch?v=g3sEsBGkLU0&list=PLTl9hO2Oobd97qfWC40gOSU8C0iu0m2l4&index=6
# Channel: CodeEmporium

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

def scaled_dot_product(q, k, v, mask=None):
    # q, k, v  = 30 x 8 x 200 x 64
    d_k = q.size()[-1] # 64
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k) # 30 x 8 x 200 x 200 dimensional tensor
    # max_seq_len x max_seq_len tensor with an additional batch dimension,
    # scaling is done to reduce variance of the product of querry and key_transpose. 
    print(f"scaled.size() : {scaled.size()}")
    if mask is not None:
        print(f"-- ADDING MASK of shape {mask.size()} --") 
        # Broadcasting add. So just the last N dimensions need to match
        # Broadcasting refers to the automatic (implicit) expansion of array dimensions to be of equal sizes,
        # without copying array data for the purpose of making arrays with different shapes have compatible shapes for element-wise operations.
        scaled += mask # 200 x 200 --> applied to every batch and every head.
    attention = F.softmax(scaled, dim=-1) # 30 x 8 x 200 x 200
    values = torch.matmul(attention, v) # 30 x 8 x 200 x 64 --> for every batch, every head, every word. 
    return values, attention

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model # 512
        self.num_heads = num_heads # 8
        self.head_dim = d_model // num_heads # 64
        self.qkv_layer = nn.Linear(d_model , 3 * d_model) # q, k, v concatenated into a single stacked tensor, computed parallely - 512 x 1536
        self.linear_layer = nn.Linear(d_model, d_model) # 512 x 512 (Feed Forward Layer)
    
    def forward(self, x, mask=None):
        batch_size, max_sequence_length, d_model = x.size() # 30 x 200 x 512
        print(f"x.size(): {x.size()}")
        qkv = self.qkv_layer(x) # 30 x 200 x 1536 (for each word we have q, k and v)
        print(f"qkv.size(): {qkv.size()}")
        # Breaking them up into 8 heads
        qkv = qkv.reshape(batch_size, max_sequence_length, self.num_heads, 3 * self.head_dim) # 30 x 200 x 8 x 192 --> per head
        print(f"qkv.size(): {qkv.size()}")
        qkv = qkv.permute(0, 2, 1, 3) # 30 x 8 x 200 x 192
        print(f"qkv.size(): {qkv.size()}")
        q, k, v = qkv.chunk(3, dim=-1) # each are 30 x 8 x 200 x 64 --> q, k, v are decomposed
        print(f"q size: {q.size()}, k size: {k.size()}, v size: {v.size()}, ")
        values, attention = scaled_dot_product(q, k, v, mask) # attention - 30 x 8 x 200 x 200, value - 30 x 8 x 200 x 64
        print(f"values.size(): {values.size()}, attention.size:{ attention.size()} ")
        values = values.reshape(batch_size, max_sequence_length, self.num_heads * self.head_dim) # 30 x 200 x (8 x 64) = 30 x 200 x 512 (shape retained)
        print(f"values.size(): {values.size()}")
        out = self.linear_layer(values)
        print(f"out.size(): {out.size()}")
        return out


class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape, eps=1e-5):
        super().__init__()
        self.parameters_shape=parameters_shape # typically your embedding dimension - 512
        self.eps=eps # to prevent division by 0
        # 2 Learnable parameters (gamma, beta)
        self.gamma = nn.Parameter(torch.ones(parameters_shape)) # [512]
        self.beta =  nn.Parameter(torch.zeros(parameters_shape)) # [512]

    def forward(self, inputs): # 30 x 200 x 512
        dims = [-(i + 1) for i in range(len(self.parameters_shape))] # [-1]
        mean = inputs.mean(dim=dims, keepdim=True) # 30 x 200 x 1
        print(f"Mean ({mean.size()})")
        var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True) # 30 x 200 x 1
        std = (var + self.eps).sqrt() # 30 x 200 x 1
        print(f"Standard Deviation  ({std.size()})")
        y = (inputs - mean) / std # 30 x 200 x 512
        print(f"y: {y.size()}")
        # Layer normalized values
        out = self.gamma * y  + self.beta # 30 x 200 x 512
        print(f"self.gamma: {self.gamma.size()}, self.beta: {self.beta.size()}")
        print(f"out: {out.size()}")
        return out

  
class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden) # 512 x 2048
        self.linear2 = nn.Linear(hidden, d_model) # 2048 x 512
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x) # 30 x 200 x 512
        print(f"x after first linear layer: {x.size()}")
        x = self.relu(x) # 30 x 200 x 2048
        print(f"x after activation: {x.size()}")
        x = self.dropout(x) # 30 x 200 x 2048
        print(f"x after dropout: {x.size()}")
        x = self.linear2(x) # 30 x 200 x 512 --> Shape Retained
        print(f"x after 2nd linear layer: {x.size()}")
        return x


class EncoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads) # performs multi-head self attention
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x):
        residual_x = x # 30 x 200 x 512
        print("------- ATTENTION 1 ------")
        x = self.attention(x, mask=None) # Encoder doesn't need a mask. Only decoder needs one. # 30 x 200 x 512
        print("------- DROPOUT 1 ------")
        x = self.dropout1(x) # 30 x 200 x 512
        print("------- ADD AND LAYER NORMALIZATION 1 ------")
        x = self.norm1(x + residual_x) # 30 x 200 x 512
        residual_x = x # 30 x 200 x 512
        print("------- ATTENTION 2 ------")
        x = self.ffn(x) # 30 x 200 x 512
        print("------- DROPOUT 2 ------")
        x = self.dropout2(x) # 30 x 200 x 512
        print("------- ADD AND LAYER NORMALIZATION 2 ------")
        x = self.norm2(x + residual_x) # 30 x 200 x 512
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers):
        super().__init__()
        self.layers = nn.Sequential(*[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob) for _ in range(num_layers)])
        # * is used to deconstruct the list 

    def forward(self, x):
        x = self.layers(x)
        return x

encoder = Encoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers)
x = torch.randn( (batch_size, max_sequence_length, d_model) ) # includes positional encoding
out = encoder(x)