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

from engine import train_one_epoch, evaluate
from newNN import get_transform
from newNN import get_model_instance_segmentation
from newNN import get_model_from_backbone
from CowDataset import CowDataset

import utils
import transforms as T
import argparse
from nms import fnms

def initArgs():
    parser = argparse.ArgumentParser("This program loads image sample and hdf5 model data to test the training performance")
    parser.add_argument('--src', default = 'D:/Learning/Postgraduate/Final_Project/Workshop/OFlow/Test2', type = str, help = 'Image(Directory) source directory')
    parser.add_argument('--mode', default = 'finetuning', type = str, help = 'PTH model type selection(finetuned or backbone-based(deprecated))')
    parser.add_argument('--model', default = 'PTH/MASK_RCNN_FT_100_preFusion.pth', type = str, help = 'PTH source directory')
    parser.add_argument('--thr', default = 0.75, type = float, help = 'threshold for result selection')
    parser.add_argument('--dest', default = 'D:\Learning\Postgraduate\Final_Project\Workshop\mAP\input\detection-results', type = str, help = 'text file location')
    args = parser.parse_args()
    return args

def dictSort(e):
    return e['scores']

def IoU(truth: torch.Tensor, prediction: torch.Tensor):
    SMOOTH = 1e-8
    pred = prediction

    X1 = pred.size()[1]
    Y1 = pred.size()[2]
    X2 = truth.size()[1]
    Y2 = truth.size()[2]

    layerI = torch.zeros((X1, Y1), dtype = torch.int32)
    layerU = torch.zeros((X2, Y2), dtype = torch.int32)

    for i in range(pred.size()[0]):
        layerI = (layerI | pred[i])

    for i in range(truth.size()[0]):
        layerU = (layerU | truth[i])

    # I = Image.fromarray(layerI.byte().cpu().numpy())
    # U = Image.fromarray(layerU.byte().cpu().numpy())
    #
    # I.show()
    # U.show()

    intersection = (layerI & layerU).float().sum()
    union = (layerI | layerU).float().sum()

    IoU = (intersection + SMOOTH) / (union + SMOOTH)

    thrIoU = torch.clamp(20 * (IoU - 0.5), 0, 10).ceil() / 10

    return IoU, thrIoU

def convertColor(image):
    im = image.convert("RGBA")

    data = np.array(im)
    red, green, blue, alpha = data.T

    # Replace red with blue... (leaves alpha values alone...)
    red_areas = (red == 255) & (blue == 0) & (green == 0)
    data[..., :-1][red_areas.T] = (0, 0, 0) # Transpose back needed

    im2 = Image.fromarray(data)
    im2.show()

def stackImages(image_list):
    background = image_list.pop(0)
    if len(image_list) >= 1:
        for img in image_list:
            background.paste(img, (0,0), img)

    return background

def main():
    args = initArgs()
    #img = Image.open(args.src).convert("RGB")
    #trans = transforms.ToTensor()
    #imgTensor = trans(img)
    num_classes = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pth = args.model
    threshold = args.thr
    dest = args.dest
    #IoU = []

    if args.src:
        dir = args.src
        testDataset = CowDataset(dir, get_transform(train = False))
        testloader = torch.utils.data.DataLoader(testDataset, batch_size = 1, shuffle=False, num_workers=1, collate_fn=utils.collate_fn)
        #for i in range(len(testDataset)):
        # img, target = testDataset[1]

        if args.mode == 'finetuning':
            model = get_model_instance_segmentation(num_classes)
        elif args.mode == 'backbone':
            model = get_model_from_backbone(num_classes)
        else:
            raise ValueError("I don't understand your mode selection?")

        model.load_state_dict(torch.load(pth))
        model.eval()
        cnt = 0
        with torch.no_grad():
            for idx, (img, target) in enumerate(testloader):
                result = []
                resultTensors = []
                resultImages = []
                resultScores = []
                resultBoxes = []
                rb = []
                c = "%07d" % cnt
                filename = os.path.join(dest,"FRAME"+c+".txt")
                f = open(filename, 'w+')
                prediction = model(img)
                target = list(target)
                # print(type(prediction))
                # print(prediction)
                # origin = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
                cap = target[0]['masks'].size()[0]

                limit = prediction[0]['masks'].size()[0]

                for i in range(limit):
                    reTe = prediction[0]['masks'][i, 0].mul(255)
                    reIm = Image.fromarray(reTe.mul(255).byte().cpu().numpy())
                    reBb = prediction[0]['boxes'][i]
                    if prediction[0]['scores'][i] >= threshold:
                        tempdict = {}
                        resultTensors.append(reTe)
                        resultScores.append(prediction[0]['scores'][i])
                        resultBoxes.append(reBb)
                        tempdict['scores'] = prediction[0]['scores'][i]
                        tempdict['masks'] = reTe
                        tempdict['boxes'] = reBb
                        result.append(tempdict)
                        #result = [Image.fromarray(prediction[0]['masks'][i, 0].mul(255).byte().cpu().numpy()) for i in range(limit)]

                for box, score in zip(resultBoxes, resultScores):
                    x1, y1, x2, y2 = box[0].item(), box[1].item(), box[2].item(), box[3].item()
                    score = score.item()
                    rb.append([x1, y1, x2, y2, score])

                newrb = np.asarray(rb)
                reb, pick = fnms(newrb, 0.5, True)

                new_resultTensors = [resultTensors[i] for i in pick]

                mask_results = [Image.fromarray(r.byte().cpu().numpy()) for r in new_resultTensors]
                concat_mask = stackImages(mask_results)
                concat_mask.save(f'resultMasks/RM{c}.png', "PNG")

                newT = torch.stack(tuple(new_resultTensors),dim = 0)
                T = newT.type(torch.IntTensor)
                iou, thriou = IoU(target[0]['masks'].mul(255), T)
                print(f"IoU is {iou:.4f}, thresholded IoU is {thriou:.4f}.")

                for x1, y1, x2, y2, score in reb:
                    x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(min(1280, x2)), int(min(720, y2))
                    f.write(f"cow {score} {x1} {y1} {x2} {y2}\n")

                print(f"Successfully write prediction to {filename}")
                f.close()
                cnt += 1
        #
        # #print(type(resultTensors))
        # #print(len(resultTensors))
        # # for t in resultTensors:
        # #     print(t.size())
        # newT = torch.stack(tuple(resultTensors),dim = 0)
        # T = newT.to(torch.int32)
        # # print(newT.size())
        # # print(target['masks'].size())
        # iou, thriou = IoU(target['masks'].mul(255), T)
        # print("IoU is {}, thresholded IoU is {}.".format(float(iou), float(thriou)))
        #     #origin.show()
        #     #for r in result:
        #         #r.show()


# if len(resultTensors) > cap:
#     resultTensors.clear()
#     sublist = result.sort(key=lambda i: i['scores'],reverse=True)[:cap]
#     resultTensors = [ s['masks'] for s in sublist ]
#
# elif len(resultTensors) < cap:
#     tenExample = resultTensors[0]
#     for _ in range(cap - len(resultTensors)):
#         resultTensors.append(torch.zeros_like(tenExample))
if __name__ == '__main__':
    main()
