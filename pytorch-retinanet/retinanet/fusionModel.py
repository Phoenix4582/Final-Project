import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo
from torchvision.ops import nms
from retinanet.utils import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes
from retinanet.anchors import Anchors
from retinanet import losses

class fusedModel(nn.Module):

    def __init__(self, num_classes, block, layers1, layers2):
        self.inplanes = 64
        super(fusedModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False) #RGB Frames#
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) #Grayscaled Motion data#
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1A = self._make_layer(block, 64, layers1[0])
        self.layer2A = self._make_layer(block, 128, layers1[1], stride=2)
        self.layer3A = self._make_layer(block, 256, layers1[2], stride=2)
        self.layer4A = self._make_layer(block, 512, layers1[3], stride=2)

        self.inplanes = 64
        self.layer1B = self._make_layer(block, 64, layers2[0])
        self.layer2B = self._make_layer(block, 128, layers2[1], stride=2)
        self.layer3B = self._make_layer(block, 256, layers2[2], stride=2)
        self.layer4B = self._make_layer(block, 512, layers2[3], stride=2)

        if block == BasicBlock:
            fpn1_sizes = [self.layer2[layers1[1] - 1].conv2.out_channels, self.layer3[layers1[2] - 1].conv2.out_channels,
                         self.layer4[layers1[3] - 1].conv2.out_channels]

            fpn2_sizes = [self.layer2[layers2[1] - 1].conv2.out_channels, self.layer3[layers2[2] - 1].conv2.out_channels,
                         self.layer4[layers2[3] - 1].conv2.out_channels]

        elif block == Bottleneck:
            fpn1_sizes = [self.layer2[layers1[1] - 1].conv3.out_channels, self.layer3[layers1[2] - 1].conv3.out_channels,
                         self.layer4[layers1[3] - 1].conv3.out_channels]

            fpn2_sizes = [self.layer2[layers2[1] - 1].conv3.out_channels, self.layer3[layers2[2] - 1].conv3.out_channels,
                         self.layer4[layers2[3] - 1].conv3.out_channels]
        else:
            raise ValueError(f"Block type {block} not understood")

        self.fpn1 = PyramidFeatures(fpn1_sizes[0], fpn1_sizes[1], fpn1_sizes[2])
        self.fpn2 = PyramidFeatures(fpn2_sizes[0], fpn2_sizes[1], fpn2_sizes[2])

        self.regressionModel = RegressionModel(256)
        self.classificationModel = ClassificationModel(256, num_classes=num_classes)

        self.anchors = Anchors()

        self.regressBoxes = BBoxTransform()

        self.clipBoxes = ClipBoxes()

        self.focalLoss = losses.FocalLoss()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01

        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)

        self.linear = nn.Conv2d(2*256, 256, kernel_size = 3, stride = 1, padding = 1)

        self.freeze_bn()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, inputs):

        if self.training:
            img_batch1, img_batch2, annotations = inputs
        else:
            img_batch1, img_batch2 = inputs

        x = self.conv1(img_batch1)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        y = self.conv2(img_batch2)
        y = self.bn2(y)
        y = self.relu2(y)
        y = self.maxpool2(y)

        x1 = self.layer1A(x)
        x2 = self.layer2A(x1)
        x3 = self.layer3A(x2)
        x4 = self.layer4A(x3)

        featuresA = self.fpn1([x2, x3, x4])

        y1 = self.layer1B(y)
        y2 = self.layer2B(y1)
        y3 = self.layer3B(y2)
        y4 = self.layer4B(y3)

        featuresB = self.fpn2([y2, y3, y4])

        features = [self.linear(torch.cat((fA, fB), dim = 1)) for fA, fB in featuresA, featuresB]
        # features = torch.cat((featuresA, featuresB), dim = 1)

        regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)

        classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)

        anchors = self.anchors(img_batch)

        if self.training:
            return self.focalLoss(classification, regression, anchors, annotations)
        else:
            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

            finalResult = [[], [], []]

            finalScores = torch.Tensor([])
            finalAnchorBoxesIndexes = torch.Tensor([]).long()
            finalAnchorBoxesCoordinates = torch.Tensor([])

            if torch.cuda.is_available():
                finalScores = finalScores.cuda()
                finalAnchorBoxesIndexes = finalAnchorBoxesIndexes.cuda()
                finalAnchorBoxesCoordinates = finalAnchorBoxesCoordinates.cuda()

            for i in range(classification.shape[2]):
                scores = torch.squeeze(classification[:, :, i])
                scores_over_thresh = (scores > 0.05)
                if scores_over_thresh.sum() == 0:
                    # no boxes to NMS, just continue
                    continue

                scores = scores[scores_over_thresh]
                anchorBoxes = torch.squeeze(transformed_anchors)
                anchorBoxes = anchorBoxes[scores_over_thresh]
                anchors_nms_idx = nms(anchorBoxes, scores, 0.5)

                finalResult[0].extend(scores[anchors_nms_idx])
                finalResult[1].extend(torch.tensor([i] * anchors_nms_idx.shape[0]))
                finalResult[2].extend(anchorBoxes[anchors_nms_idx])

                finalScores = torch.cat((finalScores, scores[anchors_nms_idx]))
                finalAnchorBoxesIndexesValue = torch.tensor([i] * anchors_nms_idx.shape[0])
                if torch.cuda.is_available():
                    finalAnchorBoxesIndexesValue = finalAnchorBoxesIndexesValue.cuda()

                finalAnchorBoxesIndexes = torch.cat((finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue))
                finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx]))

            return [finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates]

def fusionResNet50(num_classes, pretrained=False, **kwargs):
    """Constructs a fused ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 6, 3], [3, 4, 6, 3], **kwargs)
    if pretrained:
        raise ValueError("Fusion model cannot load pretrained weights")
    return model
