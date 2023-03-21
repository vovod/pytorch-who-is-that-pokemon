import torch
from torchvision import models
import matplotlib.pyplot as plt
from transform import data_transforms
from PIL import Image
import matplotlib.image as mpimg
import os

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

#Load image
files = "predict"
for file in os.listdir(files):
    test_img = files + "/" + file
    img = Image.open(test_img)
    img_s = mpimg.imread(test_img)
    img = data_transforms(img).to(device)
    img = img.unsqueeze(0)

    #Predict
    output = model(img)
    _, predicted = torch.max(output, 1)
    # print(myl[predicted])

    #Show results
    plt.imshow(img_s)
    plt.axis('off')
    label = myl[predicted]
    plt.suptitle('Model found that pokemon is: {}'.format(label), size=20)
    plt.show()