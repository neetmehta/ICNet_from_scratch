from logging import root
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from data import Cityscapes
from pspnet import PSPNet

import random

random.seed(123)
torch.manual_seed(123)
print('seed created')

ROOT = "/Cityscapes"
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_WORKERS = 2
NUM_EPOCHS = 200
CKPT_DIR = "ckpt"
RESUME = True
CKPT_PATH = "ckpt/pspnet_epoch_20.ckpt"
os.makedirs(CKPT_DIR, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


train_data = Cityscapes(root=ROOT, set_type='train')
val_data = Cityscapes(root=ROOT, set_type='val')

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
start_epoch = 0
model = PSPNet(backbone_type='resnet18').to(device)
if RESUME:
    state_dict = torch.load(CKPT_PATH)
    model.load_state_dict(state_dict["model_state_dict"])
    start_epoch = state_dict["epoch"] + 1
    loss = state_dict['loss']
    print(f"Starting training from epoch: {start_epoch-1} the loss was {loss}")


criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
print(f"Number of parameters = {sum(i.numel() for i in model.parameters())}")
print('Starting training')
for epoch in range(start_epoch, NUM_EPOCHS):
    loop = tqdm(train_loader)
    mean_loss = []
    model.train()
    for image, target, label in loop:
        image, target = image.to(device), target.to(device)
        pred = model(image)
        loss = criterion(pred, target)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_description(f"Epoch [{epoch}/{NUM_EPOCHS}]")
        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")

    print("starting validation ...")
    loop = tqdm(val_loader)
    mean_loss = []
    model.eval()
    for image, target, label in loop:
        with torch.no_grad():
            image, target = image.to(device), target.to(device)
            pred = model(image)
            loss = criterion(pred, target)
            mean_loss.append(loss.item())

            loop.set_description(f"Epoch [{epoch}/{NUM_EPOCHS}]")
            loop.set_postfix(loss=loss.item())

    print(f"Mean val loss was {sum(mean_loss)/len(mean_loss)}")

    if epoch%10==0:
        state_dict = {'epoch': epoch,
                'loss': sum(mean_loss)/len(mean_loss), 
                'model_state_dict': model.state_dict(), 
                'optimizer_state_dict': optimizer.state_dict()}

        torch.save(state_dict, os.path.join(CKPT_DIR, f"pspnet_epoch_{epoch}.ckpt"))

