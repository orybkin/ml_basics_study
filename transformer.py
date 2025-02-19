
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# class Attention(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.w_q = nn.Linear(dim, dim)
#         self.w_k = nn.Linear(dim, dim)
#         self.w_v = nn.Linear(dim, dim)
#         self.w_o = nn.Linear(dim, dim)

#     def forward(self, x):
#         q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)

#         # For each query, normalize scores along key dimension
#         out = F.softmax((q @ k.permute((0,2,1))) / np.sqrt(k.shape[-1]), dim=2) @ v
#         out = self.w_o(out)
#         return out
    
class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_length, model_dim):
        super().__init__()
        self.max_length = max_length
        self.layer = nn.Linear(max_length, model_dim)

    def forward(self, x):
        position = torch.arange(x.shape[1], dtype=torch.long, device=x.device)[None].repeat(x.shape[0], 1)
        position_onehot = F.one_hot(position, self.max_length).float()
        return self.layer(position_onehot)

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.w_q = nn.Linear(dim, dim)
        self.w_k = nn.Linear(dim, dim)
        self.w_v = nn.Linear(dim, dim)
        self.w_o = nn.Linear(dim, dim)

    def forward(self, x):
        shape = list(x.shape[:-1]) + [self.n_heads, self.head_dim]
        q, k, v = self.w_q(x).reshape(shape).permute(0, 2, 1, 3), self.w_k(x).reshape(shape).permute(0, 2, 1, 3), self.w_v(x).reshape(shape).permute(0, 2, 1, 3)
        # batch x heads x sequence x features

        # Causal masking
        q_ids = torch.arange(q.shape[-2], device=x.device)[None, :]
        mask = (q_ids <= torch.arange(q.shape[-2], device=x.device)[:, None]).float()[None, None] - 1
        mask[mask == -1] = torch.inf

        # For each query, normalize scores along key dimension
        score = (q @ k.permute((0,1,3,2))) / np.sqrt(k.shape[-1]) - mask
        out = F.softmax(score, dim=3) @ v # could also do this as einsum to avoid permuting
        # import pdb; pdb.set_trace()
        # batch x heads x sequence x features
        out = out.permute((0, 2, 1, 3)).reshape(x.shape)
        out = self.w_o(out)
        return out

class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.l1 = nn.Linear(dim, dim)
        self.l2 = nn.Linear(dim, dim)

    def forward(self, x):
        out = self.l1(x)
        out = nn.ReLU()(out)
        out = self.l2(out)
        return out

class TransformerLayer(nn.Module):

    # Ideas
    # TODO Share the input and output embedding. How?
    # TODO Add different kidns of positional embedding
    # TODO add swiglu 
    # TODO what's the initialiation

    def __init__(self, dim, n_heads):
        super().__init__()
        self.mlp = MLP(dim)
        self.attention = MultiHeadAttention(dim, n_heads)

    def forward(self, x):
        """
        x: batch x sequence x input_dim
        out: batch x sequence x input_dim
        """

        # out = self.attention(F.layer_norm(x, x.shape[-1:])) + x
        # out = self.mlp(F.layer_norm(out, x.shape[-1:])) + out
        # out, q, k, v, mask, score = self.attention(x[:, :-1])
        # out1, q1, k1, v1, mask1, score1 = self.attention(x)
        # (out - out1[:, :-1]).mean()
        # (q - q1[:, :, :-1]).mean()
        # (k - k1[:, :, :-1]).mean()
        # (v - v1[:, :, :-1]).mean()
        # (mask - mask1[:, :, :-1, :-1]).mean()
        # (score - score1[:, :, :-1, :-1]).mean()
        # out_ = F.softmax(score, dim=3) @ v
        # out_1 = F.softmax(score1, dim=3) @ v1
        # (out_ - out_1[:, :, :-1]).mean()
        # out__ = out_.permute((0, 2, 1, 3)).reshape(x[:, :-1].shape)
        # out__1 = out_1.permute((0, 2, 1, 3)).reshape(x.shape)
        # (out__ - out__1[:, :-1]).mean()
        

        out = self.attention(x) + x
        out = self.mlp(out) + out
        return out
    

class Transformer(nn.Module):
    def __init__(self, in_dim, model_dim, n_heads, n_layers, max_length=100):
        super().__init__()
        self.input_layer = nn.Linear(in_dim, model_dim)
        self.positional_embedding = LearnedPositionalEmbedding(max_length, model_dim)
        self.layers = nn.Sequential(*[TransformerLayer(model_dim, n_heads) for _ in range(n_layers-1)], nn.Linear(model_dim, in_dim))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.in_dim = in_dim

    def forward(self, x):
        out = self.input_layer(x) + self.positional_embedding(x)
        out = self.layers(out)
        return F.softmax(out, dim=-1), out

    def loss(self, x, preds, logits):
        """
        x: batch x sequence x codebook
        labels: batch x sequence - 1 x codebook
        """
        loss = F.binary_cross_entropy_with_logits(logits[:, :-1], x[:, 1:]) # Autoregressive training
        # TODO train the first token as well - starting from nothing
        return loss

    def train(self, x):
        self.optimizer.zero_grad()
        preds, logits = self(x)
        loss = self.loss(x, preds, logits)
        loss.backward()
        self.optimizer.step()
        return loss, preds

    def sample(self, x, length):
        out = []
        for i in range(length - x.shape[1]):
            out = self(x)[0][:, -1:]
            out = F.one_hot(out.argmax(-1), self.in_dim).float()
            x = torch.cat([x, out], 1)
        return x



# Arithmetic sequence dataset
batch = 10
length = 11
features = 20
data_starts = torch.randint(0, features - length + 1, (10,)) # batch
data_sequence = torch.arange(length)[None] + data_starts[:, None] # batch, length
data_onehot = F.one_hot(data_sequence, features).float().cuda() # batch, length, features

model = Transformer(in_dim=features, model_dim=6, n_heads=2, n_layers=9)
model.cuda()

for i in range(1000):
    loss, preds = model.train(data_onehot)
    print(loss)

print(data_onehot.argmax(-1))
print(model.sample(data_onehot[:, :1], length=length).argmax(-1))


import pdb; pdb.set_trace() 


torch.utils.data.TensorDxataset

import numpy as np
from deepul.hw1_helper import (
    # Q1
    visualize_q1_data,
    q1_sample_data_1,
    q1_sample_data_2,
    q1_save_results,
    # Q2
    q2a_save_results,
    q2b_save_results,
    visualize_q2a_data,
    visualize_q2b_data,
    # Q3
    q3ab_save_results,
    q3c_save_results,
    # Q4
    q4a_save_results,
    q4b_save_results,
    # Q5
    visualize_q5_data,
    q5a_save_results,
    # Q6
    visualize_q6_data,
    q6a_save_results,
)