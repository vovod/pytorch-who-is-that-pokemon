import torch
from torchvision import models
from PIL import Image
import torch.nn as nn
from torchvision import transforms
import os
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets
from transform import data_transforms


device = 'cuda'

#Load model
path="weights/Acc0.6157.pth"
TRAIN_MODE = {"pkm": 151, "pkm_t":3}
model =  models.resnet18(num_classes=151).to(device)
model.load_state_dict(torch.load(path))
model.eval()


#Read obj_names
file1 = open('obj_names.txt', 'r')
Lines = file1.readlines()
myl=[]
for line in Lines:
    string = line.strip().replace("\t","")
    for i in range(10):
        string = string.replace(str(i),'')
    myl.append(string)
myl.sort()
# print(myl)

#Load data
batch_size = 8
dataset_dir = "E:\data\pkm_classify_png"
testset = datasets.ImageFolder(root=dataset_dir, transform=data_transforms)
test_load = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)
images, labels = next(iter(test_load))
images = images.to(device)

#Predict
outputs = model(images)
_, predicted = torch.max(outputs, 1)


# Show results
for i in range(batch_size):
    plt.subplot(2, int(batch_size/2), i + 1)
    img = images[i]
    img = img / 2 + 0.5
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    color = "green"
    label = myl[predicted[i]]
    if myl[labels[i]] != myl[predicted[i]]:
        color = "red"
        label = "(" + label + ")"
    plt.title(label, color=color)
plt.suptitle('Objects Found by Model', size=20)
plt.show()