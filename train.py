import os
import torch
import time
import torch.nn as nn
from tqdm import tqdm
from torchvision import models
from preprocess import train_dataloader, val_dataloader

device = 'cuda'
os.makedirs('weights/')
load_dict = "weights/"

TRAIN_MODE = {"pkm": 151, "pkm_t":3}

# model_ft = models.resnet18(num_classes = TRAIN_MODE.get("pkm")).to(device)
model_ft = models.efficientnet_v2_s(num_classes = TRAIN_MODE.get("pkm")).to(device)

model_ft.classifier = nn.Sequential(
        nn.Dropout(p=0.5, inplace=True),
        nn.Linear(1280, TRAIN_MODE.get("pkm")),
    ).to(device)

# model_ft.load_state_dict(torch.load(""))

losses = []
accuracies = []
epoches = 100

start = time.time()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_ft.parameters(), lr=0.01)
# conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# print(device)

for epoch in range(epoches):
    epoch_loss = 0
    epoch_accuracy = 0

    for X, y in tqdm(train_dataloader):
        X = X.to(device)
        y = y.to(device)
        preds = model_ft(X).to(device)
        loss = loss_fn(preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accuracy = ((preds.argmax(dim=1)==y).float().mean())
        epoch_accuracy+=accuracy
        epoch_loss+=loss

        # print('.', end='', flush=True)
    
    epoch_accuracy = epoch_accuracy/len(train_dataloader)
    accuracies.append(epoch_accuracy)
    epoch_loss = epoch_loss/len(train_dataloader)
    losses.append(epoch_loss)

    print("\n --- Epoch: {}, train loss: {:.4f}, train acc: {:.4f}, time: {}".format(epoch, epoch_loss, epoch_accuracy, time.time() - start))

    with torch.no_grad():
        test_epoch_loss = 0
        test_epoch_accuracy = 0

        for test_X, test_y in tqdm(val_dataloader):
            test_X = test_X.to(device)
            test_y = test_y.to(device)

            test_preds = model_ft(test_X).to(device)
            test_loss = loss_fn(test_preds, test_y)

            test_epoch_loss += test_loss
            test_accuracy = ((test_preds.argmax(dim=1)==test_y).float().mean())

            test_epoch_accuracy += test_accuracy

        test_epoch_accuracy = test_epoch_accuracy/len(val_dataloader)
        test_epoch_loss = test_epoch_loss / len(val_dataloader)

        print("Epoch: {}, test loss: {:.4f}, test acc: {:.4f}, time: {}\n".format(epoch, test_epoch_loss, test_epoch_accuracy, time.time() - start))
        torch.save(model_ft.state_dict(), os.path.join(
            load_dict, "Epoch{}Acc{:.4f}.pth".format(epoch+1, test_epoch_accuracy)))