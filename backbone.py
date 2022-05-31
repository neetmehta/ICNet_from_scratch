from turtle import forward
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152 
from torchvision.models import densenet121, densenet161, densenet169, densenet201, squeezenet1_0, squeezenet1_1
import torch.nn as nn

class ResNet(nn.Module):

    def __init__(self, model_type, pretrained) -> None:
        super(ResNet, self).__init__()

        if model_type == 'resnet18':
            self.model = resnet18(pretrained)

        elif model_type == 'resnet34':
            self.model = resnet34(pretrained)

        elif model_type == 'resnet50':
            self.model = resnet50(pretrained)

        elif model_type == 'resnet101':
            self.model = resnet101(pretrained)

        elif model_type == 'resnet152':
            self.model = resnet152(pretrained)

        else:
            print("unknown model")

        del self.model._modules['avgpool']
        del self.model._modules['fc']

    def forward(self, x):

        x = self.model._modules['conv1'](x)
        x = self.model._modules['bn1'](x)
        x = self.model._modules['relu'](x)
        x = self.model._modules['maxpool'](x)
        x = self.model._modules['layer1'](x)
        x = self.model._modules['layer2'](x)
        x = self.model._modules['layer3'](x)
        x = self.model._modules['layer4'](x)
        return x
        


class DenseNet(nn.Module):

    def __init__(self, model_type, pretrained) -> None:
        super(DenseNet, self).__init__()

        if model_type == 'densenet121':
            self.model = densenet121(pretrained)

        elif model_type == 'densenet161':
            self.model = densenet161(pretrained)

        elif model_type == 'densenet169':
            self.model = densenet169(pretrained)

        elif model_type == 'densenet201':
            self.model = densenet201(pretrained)

        else:
            print("unknown model")

        del self.model._modules['classifier']

    def forward(self, x):

        return self.model._modules['features'](x)

class SqueezeNet(nn.Module):

    def __init__(self, model_type, pretrained) -> None:
        super(SqueezeNet, self).__init__()

        if model_type == 'squeezenet1_1':
            self.model = squeezenet1_1(pretrained)

        elif model_type == 'squeezenet1_0':
            self.model = squeezenet1_0(pretrained)

        else:
            print("unknown model")

        if 'classifier' in self.model._modules.keys():
            del self.model._modules['classifier']

    def forward(self, x):

        return self.model._modules['features'](x)

def model_factory(model_type, pretrained):
    if model_type == 'densenet121':
        model = DenseNet("densenet121", pretrained)

    elif model_type == 'densenet161':
        model = DenseNet("densenet161", pretrained)

    elif model_type == 'densenet169':
        model = DenseNet("densenet169", pretrained)

    elif model_type == 'densenet201':
        model = DenseNet("densenet201", pretrained)

    elif model_type == 'resnet18':
        model = ResNet("resnet18",pretrained)

    elif model_type == 'resnet34':
        model = ResNet("resnet34",pretrained)

    elif model_type == 'resnet50':
        model = ResNet("resnet50",pretrained)

    elif model_type == 'resnet101':
        model = ResNet("resnet101",pretrained)

    elif model_type == 'resnet152':
        model = ResNet("resnet152",pretrained)

    elif model_type == 'squeezenet1_1':
        model = SqueezeNet("squeezenet1_1",pretrained)

    elif model_type == 'squeezenet1_0':
        model = SqueezeNet("squeezenet1_0",pretrained)

    else:
        print("unknown model")

    return model