import torchvision.transforms.functional as TF
import random

class RandomHorizontalflip:
    """"""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, mask):
        if random.random() > self.prob:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        return image, mask

class RandomVerticalflip:
    """"""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, mask):
        if random.random() > self.prob:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        return image, mask