
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np

import math


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


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, max_length, model_dim):
        super().__init__()
        self.max_length = max_length
        self.model_dim = model_dim

    def forward(self, x):
        
        position = torch.arange(x.shape[1], dtype=torch.long, device=x.device)[None, :, None].repeat(x.shape[0], 1, 1)
        dimension = torch.arange(self.model_dim / 2, dtype=torch.long, device=x.device)[None, None, :].repeat(x.shape[0], 1, 1)

        sin = torch.sin(position/10000**(2 * dimension / self.model_dim))
        cos = torch.cos(position/10000**(2 * dimension / self.model_dim))

        # from PIL import Image
        # image = torch.cat((sin[0], cos[0]), 1).cpu().numpy()
        # image = ((image + 1)/2*255).astype(np.uint8)
        # Image.fromarray(image).save("your_file.png")
        return torch.cat([sin,cos],-1)
    

class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_length, model_dim):
        super().__init__()
        self.max_length = max_length
        self.layer = nn.Linear(max_length, model_dim)

    def forward(self, x):
        position = torch.arange(x.shape[1], dtype=torch.long, device=x.device)[None].repeat(x.shape[0], 1)
        position_onehot = F.one_hot(position, self.max_length).float()
        return self.layer(position_onehot)
    

class NoPositionalEmbedding(nn.Module):
    def __init__(self, max_length, model_dim):
        super().__init__()
        self.max_length = max_length
        self.model_dim = model_dim

    def forward(self, x):
        return torch.zeros((x.shape[0], x.shape[1], self.model_dim), dtype=torch.float, device=x.device)


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

        # batch x heads x sequence x features
        out = out.permute((0, 2, 1, 3)).reshape(x.shape)
        out = self.w_o(out)
        return out

class SwiGLUMLP(nn.Module):
    def __init__(self, model_dim, ff_dim=None):
        if ff_dim is None: ff_dim = model_dim * 8 // 3
        super().__init__()
        self.l1 = nn.Linear(model_dim, ff_dim, bias=False)
        self.l2 = nn.Linear(model_dim, ff_dim, bias=False)
        self.l3 = nn.Linear(ff_dim, model_dim, bias=False)

    def forward(self, x):
        out1 = self.l1(x)
        out2 = self.l2(x)
        out = nn.SiLU()(out1) * out2
        out = self.l2(out)
        return out
    
class MLP(nn.Module):
    def __init__(self, model_dim, ff_dim=None):
        if ff_dim is None: ff_dim = model_dim * 4 
        super().__init__()
        self.l1 = nn.Linear(model_dim, ff_dim)
        self.l2 = nn.Linear(ff_dim, model_dim)

    def forward(self, x):
        out = self.l1(x)
        # out = nn.ReLU()(out)
        out = nn.GELU()(out) # TODO make this parametric
        out = self.l2(out)
        return out

class TransformerLayer(nn.Module):

    # Ideas
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
        

        out = self.attention(F.layer_norm(x, x.shape[-1:])) + x
        out = self.mlp(F.layer_norm(out, out.shape[-1:])) + out
        return out
    

class Transformer(nn.Module):
    def __init__(self, in_dim, model_dim, n_heads, n_layers, max_iterations, max_length=100, lr_warmup_steps=1000, min_lr_multiplier=1e-2, max_lr=1e-3):
        super().__init__()
        self.input_layer = nn.Linear(in_dim, model_dim, bias=False)
        # self.positional_embedding = SinusoidalPositionalEmbedding(max_length, model_dim)
        self.positional_embedding = LearnedPositionalEmbedding(max_length, model_dim)
        # self.positional_embedding = NoPositionalEmbedding(max_length, model_dim)
        self.in_dim = in_dim
        self.layers = nn.Sequential(*[TransformerLayer(model_dim, n_heads) for _ in range(n_layers-1)])

        # lr scheduling
        self.max_iterations = max_iterations
        self.lr_warmup_steps = lr_warmup_steps
        self.warmup_lr = 1e-4
        self.max_lr = max_lr
        self.min_lr = max_lr * min_lr_multiplier
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.warmup_lr)
        self.it = 0

    def apply_output_layer(self, x, input_layer):
        out = x @ input_layer.weight
        return out

    @property
    def lr(self):
        return self.optimizer.param_groups[0]['lr']

    def lr_update(self, iteration):
        warmup_lr = self.warmup_lr
        max_lr = self.max_lr
        min_lr = self.min_lr
        lr = self.lr
        if self.it <= self.lr_warmup_steps:
            step = (max_lr - warmup_lr) / self.lr_warmup_steps
            lr = lr + step
        elif self.it <= self.max_iterations:
            max_decay_iterations = self.max_iterations - self.lr_warmup_steps
            decay_iteration = iteration - self.lr_warmup_steps
            diff = (max_lr - min_lr)
            lr = min_lr + 0.5 * diff * (1 + math.cos(decay_iteration/max_decay_iterations * math.pi))
        elif self.it > self.max_iterations:
            lr = min_lr
        self.optimizer.param_groups[0]['lr'] = lr

    def forward(self, x):
        out = self.input_layer(x) + self.positional_embedding(x)
        out = self.layers(out)
        out = self.apply_output_layer(out, self.input_layer)
        return F.softmax(out, dim=-1), out

    def loss(self, x, preds, logits):
        """
        x: batch x sequence x codebook
        labels: batch x sequence - 1 x codebook
        """
        # loss = F.binary_cross_entropy_with_logits(logits[:, :-1], x[:, 1:]) # Autoregressive training
        loss = F.cross_entropy(logits[:, :-1].flatten(0,1), x[:, 1:].flatten(0,1))
        accuracy = (logits[:, :-1].argmax(-1) == x[:, 1:].argmax(-1)).float().mean()
        return loss, accuracy

    def train(self, x):
        self.optimizer.zero_grad()
        preds, logits = self(x)
        loss, accuracy = self.loss(x, preds, logits)
        loss.backward()
        self.optimizer.step()
        self.it += 1
        self.lr_update(self.it)
        return loss, accuracy, preds

    def test(self, x):
        with torch.no_grad():
          preds, logits = self(x)
          loss, accuracy = self.loss(x, preds, logits)
        return loss, accuracy, preds

    def sample(self, x, length, deterministic=True):
        out = []
        for i in range(length - x.shape[1]):
            out = self(x)[0][:, -1:]
            if deterministic:
                out = out.argmax(-1)
            else:
                out = Categorical(out).sample()
            out = F.one_hot(out, self.in_dim).float()
            x = torch.cat([x, out], 1)
        return x


class BWQuantizeTokenizer():
    def __init__(self, n_colors):
        self.tokens = n_colors + 1
        self.bos = n_colors

    def tokenize(self, image):
        image_flatten = image.reshape([image.shape[0], -1])
        bos = torch.ones_like(image_flatten[:, :1]) * self.bos
        image_bos = torch.cat([bos, image_flatten], -1)
        image_onehot = F.one_hot(image_bos, self.tokens).float()
        return image_onehot
    
    def get_bos_tokens(self, batch_size):
        bos = torch.ones((batch_size, 1), dtype=torch.long, device='cuda') * self.bos
        bos_onehot = F.one_hot(bos, self.tokens).float()
        return bos_onehot


class ColorQuantizeTokenizer():
    def __init__(self, n_colors, image_shape):
        self.n_colors = n_colors
        self.tokens = n_colors ** 3 + 1
        self.bos = self.tokens - 1
        self.image_shape = image_shape

    def tokenize(self, image):
        image_merge_colors = image[..., 0] + image[..., 1] * self.n_colors + image[..., 2] * self.n_colors**2
        image_flatten = image_merge_colors.reshape([image_merge_colors.shape[0], -1])
        bos = torch.ones_like(image_flatten[:, :1]) * self.bos
        image_bos = torch.cat([bos, image_flatten], -1)
        image_onehot = F.one_hot(image_bos, self.tokens).float()
        return image_onehot
    
    def detokenize(self, tokens):
        tokens_index = tokens.argmax(-1)[:, 1:]
        r = tokens_index % self.n_colors
        tokens_index = (tokens_index - r) // self.n_colors
        g = tokens_index % self.n_colors
        b = (tokens_index - g) // self.n_colors
        image = torch.stack([r,g,b], -1)
        image = image.reshape([tokens.shape[0]] + list(self.image_shape))
        return image
    
    def get_bos_tokens(self, batch_size):
        bos = torch.ones((batch_size, 1), dtype=torch.long, device='cuda') * self.bos
        bos_onehot = F.one_hot(bos, self.tokens).float()
        return bos_onehot



# # Arithmetic sequence dataset
# batch = 10
# length = 11
# features = 20
# data_starts = torch.randint(0, features - length + 1, (10,)) # batch
# data_sequence = torch.arange(length)[None] + data_starts[:, None] # batch, length
# data_onehot = F.one_hot(data_sequence, features).float().cuda() # batch, length, features

# model = Transformer(in_dim=features, model_dim=6, n_heads=2, n_layers=9)
# model.cuda()

# for i in range(1000):
#     loss, preds = model.train(data_onehot)
#     print(loss)

# print(data_onehot.argmax(-1))
# print(model.sample(data_onehot[:, :1], length=length).argmax(-1))


