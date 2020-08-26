import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from xml.etree import ElementTree as et
import xml.dom.minidom as x
import math
import os

from CowDataset import CowDataset
from nnModels import ResNet50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_root = '/mnt/storage/scratch/ho19002/Data/July_22/images/train2017'
test_root = '/mnt/storage/scratch/ho19002/Data/July_22/images/val2017'
train_ann = "/mnt/storage/scratch/ho19002/Data/July_22/annotations/instances_train2017.json"
test_ann = "/mnt/storage/scratch/ho19002/Data/July_22/annotations/instances_val2017.json"
full_root = '/mnt/storage/scratch/ho19002/Data/July_24/'
#full_root = 'D:\Learning\Postgraduate\Final_Project\Workshop\OFlow\Sorted'

epochs = 20
#lr = 0.0001
num_classes = 4
batch_size = 1
in_channels = 1

TRANSFORM = True
#NN_TYPE = "CNN"

if TRANSFORM == True:
    trans = transforms.Compose([
        #transforms.ToPILImage(),
        #transforms.Resize(256,256),
        #transforms.RandomCrop((224,224)),
        #transforms.ColorJitter(brightness=0.5),
        #transforms.RandomRotation(degrees=45),
        #transforms.RandomGrayscale(p=0.2),
        #transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        #transforms.RandomVerticalFlip(p=0.1),
        #transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]) #(value - mean) / std
    ])

else:
    trans = transforms.Compose([
        transforms.ToTensor(),
    ])

train = CowDataset(root = full_root, transforms = trans)
test = CowDataset(root = full_root, transforms = transforms.ToTensor())

indices = torch.randperm(len(train)).tolist()
train = torch.utils.data.Subset(train, indices[:-30])
test = torch.utils.data.Subset(test, indices[-30:])
print('Number of training samples: ', len(train))
print('Number of testing samples: ', len(test))

# print('Number of samples: ', len(full))
#
# data, mask, target = full[188]
#
# box = []
# print(type(data))
# print(data.size())
# #data.show()
# print(type(mask))
# print(mask.size())
# #mask.show()
# print(type(target))
# print(target)
# for item in target["boxes"]:
#     box.append(item)
#
# print(box)
# print(box[0])
# print(type(box[0]))
# l = torch.tensor(np.array(box[0]))
# print(l)
# print(type(l))


trainloader = DataLoader(dataset = train, batch_size = batch_size, shuffle = True)
testloader = DataLoader(dataset = test, batch_size = batch_size, shuffle = True)

model = ResNet50(num_classes = 4).to(device)

criterion = nn.MSELoss()
params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params, lr=0.005,momentum=0.9, weight_decay=0.0005)

lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.1)

for epoch in range(epochs):
    for batch_idx, (data, mask, targets) in enumerate(trainloader):
        box = []
        # Get data to cuda if possible
        data = data.to(device=device)
        #print(data.size())
        for item in targets["boxes"]:
            box.append(item)

        # tar = torch.cat((b for b in box), dim = 1)
        # for item in targets:
        #     box.append(item.get('bbox'))
        #
        # print(box)

        t = torch.cat(tuple(box),dim = 1)
        # print(t)
        # print(t.shape)
        #ten = torch.transpose(t,0,1)
        # print(ten)
        # print(ten.size())
        tar = t.clone().detach().to(device=device)
        #print(tar.size())
        # forward
        score = model(data)
        #print(score.size())
        loss = criterion(score, tar)
        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()
    print("loss in epoch {} is {}.".format(epoch+1, float(loss)))
    #print("bnd_loss in epoch {} is {}.".format(epoch+1, float(bnd_loss)))

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    cnt = 1
    with torch.no_grad():
        for x, m, y in loader:
            ybox = []
            x = x.to(device=device)
            for item in y["boxes"]:
                ybox.append(item)

            yt = torch.cat(tuple(ybox),dim = 1)
            yTruth = yt.clone().detach().to(device=device)

            predictions = model(x)
            print(f'Prediction item No.{cnt} is {predictions}')
            # _, predictions = scores.max(1)
            threshold = criterion(predictions, yTruth)
            num_correct += (threshold <= 0.01).sum()
            num_samples += predictions.size(0)

            print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')

    model.train()

print("Checking accuracy on training data")
check_accuracy(trainloader, model)
print("Checking accuracy on testing data")
check_accuracy(testloader, model)
print("That's it!")
