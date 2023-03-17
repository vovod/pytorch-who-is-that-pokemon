import torch
import copy
import time
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from tqdm import tqdm
from torchvision import models
from datetime import date
from preprocess import train_dataloader, val_dataloader, out_dir

device = 'cuda'

TRAIN_MODE = {"pkm": 151, "pkm_t":3}

model_ft = models.efficientnet_v2_s(num_classes = TRAIN_MODE.get("pkm_t"))

model_ft.classifier = nn.Sequential(
        nn.Dropout(p=0.5, inplace=True),
        nn.Linear(1280, TRAIN_MODE.get("pkm_t")),
    )

model_ft = model_ft.to(device)

EPOCH = 30

def train_model(trn_dataloader, val_dataloader, workspace_dir, model, criterion, optimizer, scheduler, num_epochs=EPOCH):
    model = model.to(device)
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    with open(workspace_dir+"/log.txt", "a") as f:
        f.write(str(date.today())+"\n")
        print(str(date.today()))
        f.write("Start Training\n")
        print("Start Training")
    for epoch in range(num_epochs):
        with open(workspace_dir+"/log.txt", "a") as f:
            f.write(f'Epoch {epoch}/{num_epochs - 1}------------------------------------\n')
        print(f'Epoch {epoch}/{num_epochs - 1}------------------------------------\n')

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() 
                dataloader = trn_dataloader
            else:
                model.eval()   
                dataloader = val_dataloader

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs.to(device))
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloader)
            epoch_acc = running_corrects.double() / len(dataloader)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            with open(workspace_dir+"/log.txt", "a") as f:
                f.write(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}\n')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, workspace_dir+"/best.pth")
                with open(workspace_dir+"/log.txt", "a") as f:
                    f.write('Save model at epoch: ' + str(epoch) + '\n')
                print('Save model at epoch: ' + str(epoch) + '\n')
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), workspace_dir+"/best.pth")
    return model
criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.AdamW(model_ft.parameters(), lr=0.001, weight_decay=0.000001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(train_dataloader, val_dataloader, out_dir, model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                        num_epochs=EPOCH)