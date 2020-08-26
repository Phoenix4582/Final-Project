import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
import numpy as np
from CowDataset import CowDataset
from newNN import get_transform
import torchvision
import torchvision.transforms as transforms
from newNN import get_model_instance_segmentation
import utils

def printNorm(self, input, output):
    print('Inside ' + self.__class__.__name__ + ' forward')
    print('')
    print('input: ', type(input))
    print('input[0]: ', type(input[0]))
    print('output: ', type(output))
    print('')
    print('input size:', input[0].size())
    print('output size:', output.data.size())
    print('output norm:', output.data.norm())
    return input.data

class PostFusionModel(nn.Module):
    def __init__(self, modelA, modelB, num_classes):
        super(PostFusionModel, self).__init__()
        self.featuresA = modelA
        self.featuresB = modelB
        self.num_classes = num_classes

        # self.modelB.roi_heads.box_head.fc7 = nn.Identity()
        # self.modelB.roi_heads.box_predictor.cls_score = nn.Identity()
        # self.modelB.roi_heads.box_predictor.bbox_pred = nn.Identity()
        # self.modelB.roi_heads.mask_predictor.mask_fcn_logits = nn.Identity()

        # self.head_classifier = nn.Linear(1024+1024, 1024, bias = True)
        # self.score_predictor = nn.Linear(1024+1024, self.num_classes, bias = True)
        # self.box_predictor = nn.Linear(1024+1024, 4*self.num_classes, bias = True)
        # self.mask_predictor = nn.Conv2d(256+256, self.num_classes, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x1, y1, x2, y2):

        result = {}
        images1, images2 = x1, x2
        targets1, targets2 = y1, y2
        #
        # return self.featuresA(x)
        # print(type(x2))
        # images1 = x1.to(self.device)
        # targets1 = [{k: v.to(self.device) for k, v in x2.items()}]
        #
        # images2 = y1.to(self.device)
        # targets2 = [{k: v.to(self.device) for k, v in y2.items()}]

        # print(type(images1))
        # print(type(targets1[0]['boxes']))
        x = self.featuresA(images1, targets1)
        y = self.featuresB(images2, targets2)
        # labelsX = self.modelA.roi_heads.box_head.fc7.register_forward_hook(printNorm)
        # scoresX = self.modelA.roi_heads.box_predictor.cls_score.register_forward_hook(printNorm)
        # boxesX = self.modelA.roi_heads.box_predictor.bbox_pred.register_forward_hook(printNorm)
        # masksX = self.modelA.roi_heads.mask_predictor.mask_fcn_logits.register_forward_hook(printNorm)
        #
        # labelsY = self.modelB.roi_heads.box_head.fc7.register_forward_hook(printNorm)
        # scoresY = self.modelB.roi_heads.box_predictor.cls_score.register_forward_hook(printNorm)
        # boxesY = self.modelB.roi_heads.box_predictor.bbox_pred.register_forward_hook(printNorm)
        # masksY = self.modelB.roi_heads.mask_predictor.mask_fcn_logits.register_forward_hook(printNorm)
        # boxesX = x[0]['boxes']
        # labelsX = x[0]['labels']
        # scoresX = x[0]['scores']
        # masksX = x[0]['masks']
        # boxesY = y[0]['boxes']
        # labelsY = y[0]['labels']
        # scoresY = y[0]['scores']
        # masksY = y[0]['masks']
        # boxes_result = self.box_predictor(torch.cat(boxesX, boxesY))
        # labels_result = self.head_classifier(torch.cat(labelsX, labelsY))
        # scores_result = self.score_predictor(torch.cat(scoresX, scoresY))
        # mask_result = self.mask_predictor(torch.cat((masksX, masksY), dim = 1))
        #
        # result['boxes'] = boxes_result
        # result['labels'] = labels_result
        # result['scores'] = scores_result
        # result['masks'] = mask_result
        #
        # #resultlist.append(result)
        resultlist = [x, y]
        return resultlist

def main():
    trans = transforms.ToPILImage()
    layer = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = 'D:/Learning/Postgraduate/Final_Project/Workshop/OFlow/Sorted/test'
    #root = 'D:/Learning/Postgraduate/Final_Project/Workshop/OFlow/Sorted/New'
    # tds = CowDataset(root, get_transform(train = False), mode = 'preFusion', hasMotion = False)
    # img, target = tds[0]
    # print(target)
    # #im = Image.fromarray(img.mul(225).byte().cpu().numpy())
    # im = trans(img)
    # im.show()

    tds1 = CowDataset(root, get_transform(train = False), mode = 'normal')
    tds2 = CowDataset(root, get_transform(train = False), mode = 'motion')
    img1, target1 = tds1[0]
    img2, target2 = tds2[0]
    # print(img1.size())
    for i in range(img1.size()[0]):
        layer.append(img2[0])
    img2 = torch.stack(tuple(layer), dim = 0)
    img1, img2 = img1.unsqueeze(0).to(device), img2.unsqueeze(0).to(device)
    target1, target2 = [{k: v.to(device) for k, v in target1.items()}], [{k: v.to(device) for k, v in target2.items()}]

    # print(img2.size())
    # print(target1)
    # print(target2)
    # im1 = trans(img1)
    # im1.show()
    # im2 = trans(img2)
    # im2.show()
    # dl1 = torch.utils.data.DataLoader(
    #     tds1, batch_size=2, shuffle=False, num_workers=4,
    #     collate_fn=utils.collate_fn)
    #
    # dl2 = torch.utils.data.DataLoader(
    #     tds2, batch_size=2, shuffle=False, num_workers=4,
    #     collate_fn=utils.collate_fn)

    num_classes = 2
    modelA = get_model_instance_segmentation(num_classes)
    modelB = get_model_instance_segmentation(num_classes)

    fusionModel = PostFusionModel(modelA, modelB, num_classes)
    fusionModel.to(device)
    result = fusionModel(img1, target1, img2, target2)
    print(result)


if __name__ == '__main__':
    main()
