from transformer_igpt import Transformer, BWQuantizeTokenizer
# from transformer_igpt_rope import Transformer
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import wandb
import argparse
from datetime import datetime

from deepul.hw1_helper import (
    q3ab_save_results,
    q3c_save_results,
    save_training_plot,
    show_samples,
    get_data_dir,
    load_pickled_data,
    join,
)

parser = argparse.ArgumentParser()
parser.add_argument("experiment",type=str)
parser.add_argument("--dset_type",type=int,default=1)
parser.add_argument("--n_epochs", type=int, default=1,required=False)
parser.add_argument("--min_lr_multiplier", type=float, default=1e-2,required=False)
parser.add_argument("--max_lr", type=float, default=1e-3,required=False)
parser.add_argument("--lr_warmup_steps", type=int, default=1000,required=False)
parser.add_argument("--batch_size", type=int, default=32,required=False)
args = parser.parse_args()

wandb.login()
if args.dset_type == 1: experiment = 'q3ab_shapes_'
else: experiment = 'q3ab_mnist_'
run = wandb.init(entity='oleh-rybkin', project="interview_prep",name=experiment + args.experiment + str(datetime.now()),
                 config={'min_lr_multiplier': args.min_lr_multiplier, 'max_lr': args.max_lr, 'batch_size': args.batch_size})

## Setup
dset_type = args.dset_type
part = 'a'
if part == "a":
    dataset_suffix = ""
    channel = 1
elif part == "b":
    dataset_suffix = "_colored"
    channel = 3
else:
    raise Exception("Invalid part:", part, "Must be 'a' or 'b'")

data_dir = get_data_dir(1)
if dset_type == 1:
    train_data, test_data = load_pickled_data(
        join(data_dir, f"shapes{dataset_suffix}.pkl")
    )
    image_shape = (20, 20, channel)
elif dset_type == 2:
    train_data, test_data = load_pickled_data(
        join(data_dir, f"mnist{dataset_suffix}.pkl")
    )
    image_shape = (28, 28, channel)
else:
    raise Exception()


## Training

# 3 tokens - 0, 1, or <bos>
image_length = image_shape[0] * image_shape[1] + 1
n_epochs = args.n_epochs
batch_size = args.batch_size
n_colors = 2

tokenizer = BWQuantizeTokenizer(n_colors)

# print(train_data.dtype)
# print(torch.Tensor(train_data).cuda().dtype)
dataset = torch.utils.data.TensorDataset(torch.Tensor(train_data).cuda().long())
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_dataset = torch.utils.data.TensorDataset(torch.Tensor(test_data).cuda().long())
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

model = Transformer(in_dim = tokenizer.tokens, model_dim=128, n_heads=4,n_layers=2,
                                     max_length=image_length,max_iterations=n_epochs * len(dataloader),
                                     min_lr_multiplier=args.min_lr_multiplier, lr_warmup_steps=args.lr_warmup_steps, max_lr=args.max_lr)
model.cuda()
# (train_image,) = next(iter(dataloader))

train_losses = []
test_losses = []
for i in range(n_epochs):
  for (image,) in dataloader:
    # for j in range(len(dataloader)):
    # image = train_image
    image_onehot = tokenizer.tokenize(image)
    loss, accuracy, preds = model.train(image_onehot)
    train_losses.append(loss.cpu().item())
    wandb.log({'train_loss': loss.cpu().item(), 'train_accuracy': accuracy.cpu().item(), 'lr': model.lr,  'iteration': model.it})

  test_loss_sum = 0
  test_i = 0
  for (image,) in test_dataloader:
    image_onehot = tokenizer.tokenize(image)
    loss, accuracy, preds = model.test(image_onehot)
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
samples = model.sample(bos_onehot, length=image_length, deterministic=False).argmax(-1)[:, 1:]
samples = samples.reshape([batch_size] + list(image_shape)).cpu().float().numpy()
samples = samples.astype("float32") / channel * 255

samples_grid = torchvision.utils.make_grid(torch.Tensor(samples).permute(0,3,1,2)).permute(1,2,0).numpy().astype(np.uint8)
wandb.log({'image_samples': wandb.Image(samples_grid)})
image_grid = torchvision.utils.make_grid(torch.Tensor(image).permute(0,3,1,2)).permute(1,2,0).cpu().numpy().astype(np.uint8)
wandb.log({'image_data': wandb.Image(image_grid)})

# samples = model.sample(image_onehot, length=image_length).argmax(-1)[:, 1:]
# samples = samples.reshape([32] + list(image_shape)).cpu().float()
# samples = samples[:1].repeat(100, 1, 1, 1).numpy()

# image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
# wandb.log({"example_image": wandb.Image(image)})

## Printing
print(f"Final Test Loss: {test_losses[-1]:.4f}")
save_training_plot(
    train_losses,
    test_losses,
    f"Q3({part}) Dataset {dset_type} Train Plot",
    f"results/q3_{part}_dset{dset_type}_train_plot.png",
)
show_samples(samples, f"results/q3_{part}_dset{dset_type}_samples.png")