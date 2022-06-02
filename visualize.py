import torch
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt

from data import Cityscapes
from pspnet import PSPNet
from utils import plot_image

BACKBONE = 'resnet50'
ROOT = r"E:\Deep Learning Projects\datasets\Cityscapes"
BATCH_SIZE = 1
CKPT_DIR = "ckpt"
CKPT_PATH = "ckpt/pspnet_resnet50_epoch_70.ckpt"
OUT_DIR = f"results/{BACKBONE}"
os.makedirs(OUT_DIR, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'



models = {
            'resnet18':PSPNet(backbone_type="resnet18", backbone_out_features=512, pretrained=False),
            'resnet34':PSPNet(backbone_type="resnet34", backbone_out_features=512, pretrained=False),
            'resnet50':PSPNet(backbone_type="resnet50", backbone_out_features=2048, pretrained=False),
            'resnet101':PSPNet(backbone_type="resnet101", backbone_out_features=2048, pretrained=False),
            'resnet152':PSPNet(backbone_type="resnet152", backbone_out_features=2048, pretrained=False),
}

test_data = Cityscapes(root=ROOT, set_type='val')
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

model = models[BACKBONE].to(device)
state_dict = torch.load(CKPT_PATH)
model.load_state_dict(state_dict["model_state_dict"])

for i, (image, target, label) in enumerate(test_loader):
    
    with torch.no_grad():
        image = image.to(device)
        pred = model(image)
        pred = pred.cpu()
        fig = plot_image(pred, target, image.cpu())
        fig.savefig(os.path.join(OUT_DIR,f"visualize_{i}.png"), dpi=200)
        if i==10:
            break