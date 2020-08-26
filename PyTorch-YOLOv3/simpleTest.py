import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from torch.autograd import Variable

def main():
    motion = torch.randn([1,360,540])
    if len(motion.shape) != 3:
        motion = motion.unsqueeze(0)
        motion = motion.expand((3, motion.shape[1:]))
    else:
        motion = torch.cat((motion,motion,motion), dim = 0)
    print(motion.shape)

def subsub():
    a = torch.tensor(float('nan'))
    print(a)
    b = a!=a
    print(b)
    print(torch.sum(b))

def zero():
    a = torch.randn([0,1])
    print(a)
    print(torch.sum(a))
    print(a.sum())

def linear():
    a = torch.randn([2,3,4])
    layer = nn.Linear(4, 2)
    b = layer(a)
    print(f"input is {a}")
    print(f"output is {b}")

def sub():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config_dir = 'config/uni.data'
    data_config = parse_data_config(data_config_dir)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    dataset = ListDataset(train_path, augment=True, multiscale=True)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )
    model_def = 'config/yolov3-1cls-pre.cfg'
    model = Darknet(model_def).to(device)
    model.apply(weights_init_normal)

    for batch_i, (_, imgs, targets) in enumerate(dataloader):
        imgs = Variable(imgs.to(device))
        targets = Variable(targets.to(device), requires_grad=False)

        loss, outputs = model(imgs, targets)
        print(type(loss))
        print(type(outputs))

if __name__ == '__main__':
    sub()
