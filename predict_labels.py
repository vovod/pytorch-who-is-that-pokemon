import torch
from torchvision import models
from PIL import Image
import torch.nn as nn
from torchvision import transforms
import os
from preprocess import train_data, train_dataset, val_dataloader, train_dataloader
import matplotlib.pyplot as plt
import numpy as np

path="F:\pro_ai\dog-cat-classify - Copy\Epoch19Acc0.6157.pth"
TRAIN_MODE = {"pkm": 151, "pkm_t":3}

device = 'cuda'

model =  models.resnet18(num_classes=151).to(device)

model.load_state_dict(torch.load(path))

train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0]
npimg = img.cpu().numpy()
plt.imshow(np.transpose(npimg, (1, 2, 0)))
label = train_labels[0]
plt.show()
print(f"Label: {label}")

# source = 1

# data_transforms = transforms.Compose([
#     transforms.Resize((240, 240)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],
#                         [0.229, 0.224, 0.225])
# ])

# #read_obj_names
# file1 = open('obj_names.txt', 'r')
# Lines = file1.readlines()
# myl=[]
# for line in Lines:
#     string = line.strip().replace("\t","")
#     for i in range(10):
#         string = string.replace(str(i),'')
#     myl.append(string)


# img_path = "E:\\data\\pkm_c_aug_new\\" + myl[source]

# for file in os.listdir(img_path): 
#     img = Image.open("{}/{}".format(img_path, file))
#     transform = data_transforms
#     input = transform(img).to(device)
#     test1 = input.unsqueeze(0)
#     model.eval()

#     output = model(test1)

#     output = nn.Sigmoid()(output)
#     _,pred = output.max(1)

#     print(myl[source], myl[pred[0]], sep=" ", end="\n")

