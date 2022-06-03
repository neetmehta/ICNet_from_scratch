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


APPLY_AUG = True
PRETRAINED = True
BACKBONE = 'resnet152'
ROOT = r"/Cityscapes"
BATCH_SIZE = 8
LEARNING_RATE = 1e-5
NUM_WORKERS = 2
PIN_MEMORY = True
NUM_EPOCHS = 200
CKPT_DIR = "ckpt"
RESUME = True
CKPT_PATH = "ckpt/pspnet_resnet152_epoch_50.ckpt"
os.makedirs(CKPT_DIR, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
models = {
            'resnet18':PSPNet(backbone_type="resnet18", backbone_out_features=512, pretrained=PRETRAINED),
            'resnet34':PSPNet(backbone_type="resnet34", backbone_out_features=512, pretrained=PRETRAINED),
            'resnet50':PSPNet(backbone_type="resnet50", backbone_out_features=2048, pretrained=PRETRAINED),
            'resnet101':PSPNet(backbone_type="resnet101", backbone_out_features=2048, pretrained=PRETRAINED),
            'resnet152':PSPNet(backbone_type="resnet152", backbone_out_features=2048, pretrained=PRETRAINED),
}

train_data = Cityscapes(root=ROOT, set_type='train', apply_aug=APPLY_AUG)
val_data = Cityscapes(root=ROOT, set_type='val')

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
start_epoch = 0
model = models[BACKBONE]
val_loss = 10
if RESUME:
    state_dict = torch.load(CKPT_PATH)
    model.load_state_dict(state_dict["model_state_dict"])
    start_epoch = state_dict["epoch"] + 1
    val_loss = state_dict['loss']
    del state_dict
    model = model.to(device)
    print(f"Starting training from epoch: {start_epoch-1} the loss was {val_loss}")


criterion = torch.nn.NLLLoss2d()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
print(f"Number of parameters = {sum(i.numel() for i in model.parameters())}")
print('Starting training')
for epoch in range(start_epoch, NUM_EPOCHS):
    loop = tqdm(train_loader)
    mean_loss = []
    model.train()
    for image, target in loop:
        target = torch.argmax(target, dim=1)
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
    for image, target in loop:
        with torch.no_grad():
            target = torch.argmax(target, dim=1)
            image, target = image.to(device), target.to(device)
            pred = model(image)
            loss = criterion(pred, target)
            mean_loss.append(loss.item())

            loop.set_description(f"Epoch [{epoch}/{NUM_EPOCHS}]")
            loop.set_postfix(loss=loss.item())

    loss = sum(mean_loss)/len(mean_loss)
    print(f"Mean val loss was {loss}")

    if val_loss>loss or epoch%10==0:
        state_dict = {'epoch': epoch,
                'loss': sum(mean_loss)/len(mean_loss), 
                'model_state_dict': model.state_dict(), 
                'optimizer_state_dict': optimizer.state_dict()}
        val_loss = loss
        torch.save(state_dict, os.path.join(CKPT_DIR, f"pspnet_{BACKBONE}_epoch_{epoch}.ckpt"))

