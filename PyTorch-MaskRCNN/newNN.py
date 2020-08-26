# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import os, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from engine import train_one_epoch, evaluate
from zip_engine import train_one_epoch_zip, evaluate_zip

import utils
import transforms as T

from CowDataset import CowDataset

def get_model_from_backbone(num_classes):
    # load a pre-trained model for classification and return
    # only the features
    backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    # FasterRCNN needs to know the number of
    # output channels in a backbone. For mobilenet_v2, it's 1280
    # so we need to add it here
    backbone.out_channels = 1280

    # let's make the RPN generate 5 x 3 anchors per spatial
    # location, with 5 different sizes and 3 different aspect
    # ratios. We have a Tuple[Tuple[int]] because each feature
    # map could potentially have different sizes and
    # aspect ratios
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                    aspect_ratios=((0.5, 1.0, 2.0),))

    # let's define what are the feature maps that we will
    # use to perform the region of interest cropping, as well as
    # the size of the crop after rescaling.
    # if your backbone returns a Tensor, featmap_names is expected to
    # be [0]. More generally, the backbone should return an
    # OrderedDict[Tensor], and in featmap_names you can choose which
    # feature maps to use.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                output_size=7,
                                                sampling_ratio=2)

    # put the pieces together inside a FasterRCNN model
    model = FasterRCNN(backbone,
                        num_classes=2,
                        rpn_anchor_generator=anchor_generator,
                        box_roi_pool=roi_pooler)

    return model

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

# class fusionModel(nn.Module):
#     def __init__(self, num_classes = 2, mode = 'test'):
#         super(fusionModel, self).__init__()
#         self.modelA = get_model_instance_segmentation(num_classes)
#         self.modelB = get_model_instance_segmentation(num_classes)
#         self.mode = mode
#         self.classifier = nn.Linear(2*num_classes, num_classes)
#         # self.box_classifier = nn.Linear(2*num_classes, num_classes)
#         # self.mask_classifier = nn.Linear()
#         # self.score_classifier = nn.Linear()
#         # self.label_classifier = nn.Linear()
#
#     def forward(self, x, y):
#         if self.mode == 'train':
#             loss_x = self.modelA(x, t)
#             loss_y = self.modelA(y, t)
#             loss_x_sum = sum(loss for loss in loss_x)
#             loss_y_sum = sum(loss for loss in loss_y)
#
#             return sum(loss_x_sum, loss_y_sum)
#
#         else:
#             with torch.no_grad():
#                 x_box, y_box = [], []
#                 x_mask, y_mask = [], []
#                 x_score, y_score = [], []
#                 x_label, y_label = [], []
#                 self.modelA.eval()
#                 self.modelB.eval()
#                 t_x = self.modelA(x)
#                 targets = [{k: v for k, v in t.items()} for t in t_x]
#                 for target in targets:
#                     for k, v in target.items():
#
#                 t_y = self.modelB(y)
#                 result = self.classifier(F.relu(torch.stack((t_x, t_y), dim = 0)))
#                 return result

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    full_root = '/mnt/storage/scratch/ho19002/Data/July_26/'
    save_root = '/mnt/storage/scratch/ho19002/Data/July_26/HDF5'
    #full_root = 'D:\Learning\Postgraduate\Final_Project\Workshop\OFlow\Screenshots'
    epochs = 200
    mode = 'postFusion'
    #lr = 0.0001
    # our dataset has two classes only - background and cow
    num_classes = 2
    # batch_size = 1
    # in_channels = 1

    # use our dataset and defined transformations
    trainset = CowDataset(full_root, get_transform(train=True), mode = mode)
    # data, target = trainset[0]
    # print(data.size())
    # print(target["boxes"])
    # print(target["masks"])
    # print(torch.sum(target["masks"]))
    testset = CowDataset(full_root, get_transform(train=False), mode = mode)

    # split the dataset in train and test set
    indices = torch.randperm(len(trainset)).tolist()
    trainset = torch.utils.data.Subset(trainset, indices[:-23])
    testset = torch.utils.data.Subset(testset, indices[-23:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        trainset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        testset, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes, mode = mode)
    #model = get_model_from_backbone(num_classes)
    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    for epoch in range(epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        if (epoch+1) % 10 == 0:
            hdf5_filename = "MASK_RCNN_FT_{}.h5".format(epoch+1)
            torch.save(model.state_dict(), os.path.join(save_root, hdf5_filename))
        evaluate(model, data_loader_test, device=device, mode = mode)

    print("That's it!")

if __name__ == "__main__":
    main()
