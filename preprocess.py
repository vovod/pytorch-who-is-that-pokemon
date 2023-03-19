import torch
from torchvision import datasets
import os
from datetime import date
from transform import data_transforms


# dataset_dir = "E:/data/pokemon_classify_png"
dataset_dir = "E:/data/pkm_c_aug_new"

# out_dir = "F:/pro_ai/pkm_classify/"+'logs/pkm/effnetv2s/'+ str(date.today())
out_dir = "F:/pro_ai/pkm_c_t/"+'logs/pkm/effnetv2s/'+ str(date.today())

log_dir = out_dir + '/log.txt'
os.makedirs(out_dir, exist_ok=True)

bsz = 8

#split_data
train_dataset = datasets.ImageFolder(root=dataset_dir, transform=data_transforms)
train_data, val_data = torch.utils.data.random_split(train_dataset, [int(train_dataset.__len__()*0.8), train_dataset.__len__()-int(train_dataset.__len__()*0.8)])
#load_data
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=bsz, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=bsz, shuffle=False)

