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

class fusionModel(nn.Module):
    def __init__(self, cfg1_path, cfg2_path, num_classes, device, alpha = 0.5, beta = 0.5):
        super(fusionModel, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.modelA = Darknet(cfg1_path).to(device)
        self.modelB = Darknet(cfg2_path).to(device)
        self.modelA.apply(weights_init_normal)
        self.modelB.apply(weights_init_normal)

        self.classifier = nn.Linear((5+self.num.classes)*2, 5+self.num_classes).to('cpu')
        # self.box_classifier = nn.Linear(4*2, 4).to('cpu')
        # self.score_classifier = nn.Linear(2, 1).to('cpu')
        # self.class_classifier = nn.Linear(2*self.num_classes, self.num_classes).to('cpu')

    def forward(self, img, motion, targets = None):

        if targets is None:
            output1 = self.modelA(img, targets)
            output2 = self.modelB(motion, targets)

            output = self.classifier(F.relu(torch.cat((output1, output2), dim = 2)))
            # box1, box2 = output1[..., :4], output2[..., :4]
            # score1, score2 = torch.narrow(score1, 2, 4, 1), torch.narrow(score2, 2, 4, 1)
            # class1, class2 = output1[..., 5:], output2[..., 5:]
            #
            # box_out = self.box_classifier(F.relu(torch.cat((box1, box2), dim = 2)))
            # score_out = self.score_classifier(F.relu(torch.cat((score1, score2), dim = 2)))
            # class_out = self.class_classifier(F.relu(torch.cat((class1, class2), dim = 2)))
            # output = torch.cat((box_out, score_out, class_out), dim = 2)
            return output

        else:
            loss1, output1 = self.modelA(img, targets)
            loss2, output2 = self.modelB(motion, targets)

            loss = self.alpha * loss1 + self.beta * loss2

            output = self.classifier(F.relu(torch.cat((output1, output2), dim = 2)))
            # box1, box2 = output1[..., :4], output2[..., :4]
            # score1, score2 = output1[..., 4], output2[..., 4]
            # class1, class2 = output1[..., 5:], output2[..., 5:]
            #
            # box_out = self.box_classifier(F.relu(torch.cat((box1, box2), dim = 2)))
            # score_out = self.score_classifier(F.relu(torch.cat((score1, score2), dim = 2)))
            # class_out = self.class_classifier(F.relu(torch.cat((class1, class2), dim = 2)))
            # output = torch.cat((box_out, score_out, class_out), dim = 2)

            return loss, output
