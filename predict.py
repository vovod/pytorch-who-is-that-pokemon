import torch
from torchvision import models
from transform import data_transforms
from PIL import Image
import torch.nn as nn
from torchvision import transforms
import os

device = 'cuda'

path="F:/pro_ai/pkm_c_t/logs/pkm/effnetv2s/2023-03-17/best.pth"
TRAIN_MODE = {"pkm": 151, "pkm_t":3}
classes=['b','c','s']

model =  models.efficientnet_v2_s(num_classes=3).to(device)

model.load_state_dict(torch.load(path))
source = 0
img_path = "E:/data/pkm_c_t/0"

for file in os.listdir(img_path): 
    img = Image.open("{}/{}".format(img_path, file))
    transform = data_transforms.get("test")
    input = transform(img).to(device)
    test1 = input.unsqueeze(0)
    # print(test1.size())
    model.eval()

    output = model(test1)

    _,pred = output.max(1)

    print(classes[int(source/3)], classes[pred[0]], sep=" ", end="\n")


