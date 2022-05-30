from logging import root
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from data import Cityscapes
from pspnet import PSPNet

ROOT = r"E:\Deep Learning Projects\datasets\Cityscapes"
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
NUM_WORKERS = 0
NUM_EPOCHS = 200
CKPT_DIR = "ckpt"


device = 'cuda' if torch.cuda.is_available() else 'cpu'


train_data = Cityscapes(root=ROOT, set_type='train')
val_data = Cityscapes(root=ROOT, set_type='val')

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

model = PSPNet(backbone_type='resnet18').to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
print(f"Number of parameters = {sum(i.numel() for i in model.parameters())}")
print('Starting training')
for epoch in range(NUM_EPOCHS):
    loop = tqdm(train_loader)
    mean_loss = []
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
    for image, target, label in loop:
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

