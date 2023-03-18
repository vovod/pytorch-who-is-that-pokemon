import torch
from torchvision import models
from PIL import Image
import torch.nn as nn
from torchvision import transforms
import os

path="F:\pro_ai\dog-cat-classify - Copy\Epoch20Acc0.6914.pth"
TRAIN_MODE = {"pkm": 151, "pkm_t":3}

device = 'cuda'

model =  models.resnet18(num_classes=151).to(device)

model.load_state_dict(torch.load(path))
source = 100
img_path = "E:\\data\\pkm_c_aug\\" + str(source)

data_transforms = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
])


path = "E:\\data\\pokemon_classify\\"

file1 = open('obj_names.txt', 'r')
Lines = file1.readlines()
  
# Strips the newline character
myl=[]
for line in Lines:
    string = line.strip().replace("\t","")
    for i in range(10):
        string = string.replace(str(i),'')
    myl.append(string)

for file in os.listdir(img_path): 
    img = Image.open("{}/{}".format(img_path, file))
    transform = data_transforms
    input = transform(img).to(device)
    test1 = input.unsqueeze(0)
    model.eval()

    output = model(test1)

    prob = nn.Sigmoid()(output)
    _,pred = output.max(1)

    print(file + " class: " + myl[source] + " predict: " + myl[pred[0]])