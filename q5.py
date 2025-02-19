from transformer_igpt import Transformer, ColorQuantizeTokenizer
# from transformer_igpt_rope import Transformer
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import wandb
import argparse
from datetime import datetime
from functools import reduce

from deepul.hw1_helper import (
    q3ab_save_results,
    q5a_save_results,
    save_training_plot,
    show_samples,
    get_data_dir,
    load_pickled_data,
    join,
    load_text_data,
    save_text_to_plot,
)

parser = argparse.ArgumentParser()
parser.add_argument("experiment",type=str)
parser.add_argument("--n_epochs", type=int, default=1,required=False)
parser.add_argument("--min_lr_multiplier", type=float, default=1e-2,required=False)
parser.add_argument("--max_lr", type=float, default=1e-3,required=False)
parser.add_argument("--lr_warmup_steps", type=int, default=1000,required=False)
parser.add_argument("--batch_size", type=int, default=16,required=False)
args = parser.parse_args()

wandb.login()
run = wandb.init(entity='oleh-rybkin', project="interview_prep",name='q5_' + args.experiment + str(datetime.now()),
                 config={'min_lr_multiplier': args.min_lr_multiplier, 'max_lr': args.max_lr, 'batch_size': args.batch_size})

## Setup
dir_path = get_data_dir(1)
train_data, test_data = load_text_data(join(dir_path, "poetry.pkl"))
n_epochs = args.n_epochs
batch_size = args.batch_size
max_length = 128

class CharTokenizer():
    def __init__(self, text):
        self.text = ''.join(text)
        self.chars = np.concatenate([np.unique(list(self.text)), ['<eos>']], -1)
        self.tokens = len(self.chars)
        self.eos = self.tokens - 1

    def tokenize(self, x, full_sequence=True):
        tokens = (np.array(list(x))[:, None] == self.chars).argmax(-1)
        if full_sequence:
           tokens = np.concatenate([[self.eos], tokens, [self.eos]], -1) # Add eos tokens at beginning and end
        tokens_onehot = F.one_hot(torch.Tensor(tokens).cuda().long(), self.tokens).float()
        return tokens_onehot
    
    def detokenize(self, x):
        idxs = x.argmax(-1).cpu().numpy()
        idxs_filtered = []
        for sequence in idxs:
            if sequence[1:].max() == self.eos: sequence = sequence[:sequence[1:].argmax()+2]
            idxs_filtered.append(sequence)
            print(len(sequence))
        text = [self.chars[sequence] for sequence in idxs]
        text = [''.join(seq) for seq in text]
        return text

    def get_bos_tokens(self, batch_size):
        bos = torch.ones((batch_size, 1), dtype=torch.long, device='cuda') * self.eos
        bos_onehot = F.one_hot(bos, self.tokens).float()
        return bos_onehot

class TextDataset():
    def __init__(self, text, sequence_length):
        self.text = text
        self.sequence_length = sequence_length

    def __len__(self):
        return np.sum(list([len(x) for x in self.text])) // self.sequence_length
    
    def __getitem__(self, i):
       seq_idx = np.random.randint(len(self.text))
       sequence = self.text[seq_idx]
       sub_idx = np.random.randint(max(1, len(sequence) - self.sequence_length + 1))
       subseq = sequence[sub_idx : sub_idx + self.sequence_length]
       return subseq

# TODO add bos and eos tokens
# TODO pad with correct tokens

tokenizer = CharTokenizer(train_data + test_data)
collate = lambda x: torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=0.0, padding_side='right')

train_dataset = TextDataset([tokenizer.tokenize(x) for x in train_data], max_length)
dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate)
test_dataset = TextDataset([tokenizer.tokenize(x) for x in test_data], max_length)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

model = Transformer(in_dim=tokenizer.tokens, model_dim=128, n_heads=4, n_layers=2,
                    max_length=max_length, max_iterations=n_epochs * len(dataloader),
                    min_lr_multiplier=args.min_lr_multiplier, lr_warmup_steps=args.lr_warmup_steps, max_lr=args.max_lr)
model.cuda()

train_losses = []
test_losses = []
for i in range(n_epochs):
  for batch in dataloader:
    loss, accuracy, preds = model.train(batch)
    train_losses.append(loss.cpu().item())
    wandb.log({'train_loss': loss.cpu().item(), 'train_accuracy': accuracy.cpu().item(), 'lr': model.lr,  'iteration': model.it})

  test_loss_sum = 0
  test_i = 0
  for batch in test_dataloader:
    loss, accuracy, preds = model.test(batch)
    test_i += 1
    test_loss_sum += loss.cpu().item()
  test_losses.append(test_loss_sum / test_i)
  wandb.log({'test_loss': loss.cpu().item(), 'test_accuracy': accuracy.cpu().item(), 'iteration': model.it})

import matplotlib.pyplot as plt
plt.figure()
plt.plot(np.arange(len(train_losses)),train_losses, label='train')
plt.plot(np.arange(len(test_losses)) * len(dataloader), test_losses, label='test')
plt.yscale('log')
plt.show()

bos_onehot = tokenizer.get_bos_tokens(batch_size)
samples = model.sample(bos_onehot, length=max_length, deterministic=False)
text_samples = tokenizer.detokenize(samples)

table = wandb.Table(['text'])
for sample in text_samples: table.add_data(sample)
wandb.log({'text_samples': table})

table = wandb.Table(['text'])
for sample in tokenizer.detokenize(batch): table.add_data(sample)
wandb.log({'text_data': table})

print(f"Final Test Loss: {test_losses[-1]:.4f}")
save_training_plot(
    train_losses,
    test_losses,
    f"Q5(a) Dataset Poetry Train Plot",
    f"results/q5_a_train_plot.png",
)
for idx, txt in enumerate(text_samples):
    print(f"Sample {idx+1}\n{txt}\n")
for idx, txt in enumerate(tokenizer.detokenize(batch)):
    print(f"Data {idx+1}\n{txt}\n")
save_text_to_plot(text_samples, f"results/q5_a_samples.png")