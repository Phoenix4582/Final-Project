import torch
import torch.nn.functional as F
import numpy as np


def horisontal_flip(images, targets):
    images = torch.flip(images, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    return images, targets

def zip_horizontal_flip(images, motion, targets):
    images = torch.flip(images, [-1])
    motion = torch.flip(motion, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    return images, motion, targets
