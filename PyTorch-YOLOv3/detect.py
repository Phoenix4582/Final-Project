from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse
import numpy as np
import sklearn
import sklearn.metrics
from PIL import Image
import statistics

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator


def get_gt(path):
    gt = Image.open(path)
    gt = np.array(gt)
    obj_ids = np.unique(gt)
    obj_ids = obj_ids[1:]

    masks = gt == obj_ids[:,None,None]

    num_objs = len(obj_ids)
    boxes = []
    for i in range(num_objs):
        pos = np.where(masks[i])
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        boxes.append([xmin, ymin, xmax, ymax])

    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    labels = torch.zeros((len(boxes),), dtype=torch.float32).unsqueeze(1)
    scores = torch.ones((len(boxes),), dtype=torch.float32).unsqueeze(1)
    # print(boxes.shape)
    # print(labels.shape)
    # print(scores.shape)
    return boxes, labels, scores

# Since there is only one class, no other calculations relating to class specification is required
def IoU(w, h, gt_boxes, pred_boxes):
    SMOOTH = 1e-8
    base1 = torch.zeros((w,h),dtype=torch.int32)
    base2 = torch.zeros((w,h),dtype=torch.int32)
    for gt, pred in zip(gt_boxes, pred_boxes):
        x1_gt, y1_gt, x2_gt, y2_gt = gt.type(torch.IntTensor)
        x1_p, y1_p, x2_p, y2_p = pred.type(torch.IntTensor)
        x1_gt, y1_gt, x1_p, y1_p = max(0, x1_gt), max(0, y1_gt), max(0, x1_p), max(0, y1_p)
        x2_gt, y2_gt, x2_p, y2_p = min(x2_gt, w), min(y2_gt, h), min(x2_p, w), min(y2_p, h)
        cast1 = torch.ones(((x2_gt - x1_gt), (y2_gt - y1_gt)), dtype=torch.int32)
        cast2 = torch.ones(((x2_p - x1_p), (y2_p - y1_p)), dtype=torch.int32)
        # print(f"{y2_p},{y1_p}")
        # print(cast2.shape)
        # #print(base1[x1_gt:x2_gt, y1_gt:y2_gt])
        # print(base2[x1_p:x2_p, y1_p:y2_p].shape)
        base1[x1_gt:x2_gt, y1_gt:y2_gt] = cast1
        base2[x1_p:x2_p, y1_p:y2_p] = cast2

    intersection = (base1 & base2).float().sum()
    union = (base1 | base2).float().sum()
    IoU = (intersection + SMOOTH) / (union + SMOOTH)
    thrIoU = torch.clamp(20 * (IoU - 0.5), 0, 10).ceil() / 10
    return IoU, thrIoU

def mAP(gt, pred):
    AP = [sklearn.metrics.average_precision_score(t, p) for t, p in zip(gt, pred)]
    mAP = statistics.mean(AP)
    return mAP

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    parser.add_argument("--mode", type=str, default = 'normal', help="detection mode selection (normal or fusion)")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loc = "D:\Learning\Postgraduate\Final_Project\Workshop\mAP\input\detection-results"
    os.makedirs("output", exist_ok=True)

    if opt.mode == 'normal':
        # Set up model
        image_src = os.path.join(opt.image_folder, 'Frames')
        model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

        if opt.weights_path.endswith(".weights"):
            # Load darknet weights
            model.load_darknet_weights(opt.weights_path)
        else:
            # Load checkpoint weights
            model.load_state_dict(torch.load(opt.weights_path))

        model.eval()  # Set in evaluation mode

        dataloader = DataLoader(
            ImageFolder(image_src, img_size=opt.img_size),
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_cpu,
        )

        classes = load_classes(opt.class_path)  # Extracts class labels from file

        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        imgs = []  # Stores image paths
        img_detections = []  # Stores detections for each image index

        print("\nPerforming object detection:")
        prev_time = time.time()
        for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
            # Configure input
            input_imgs = Variable(input_imgs.type(Tensor))

            # Get detections
            with torch.no_grad():
                detections = model(input_imgs)
                detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

            # Log progress
            current_time = time.time()
            inference_time = datetime.timedelta(seconds=current_time - prev_time)
            prev_time = current_time
            print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

            # Save image and detections
            imgs.extend(img_paths)
            img_detections.extend(detections)

        # Bounding-box colors
        cmap = plt.get_cmap("tab20b")
        colors = [cmap(i) for i in np.linspace(0, 1, 20)]

        cnt = 0
        print("\nSaving images:")
        # Iterate through images and save plot of detections
        for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
            c = "%06d" % cnt
            print("(%d) Image: '%s'" % (img_i, path))
            # Get file name of an image and prepare prediction txt file
            img_name = path.split("\\")[-1]
            # print(img_name)
            txt_name = img_name.replace(".png",".txt")
            f = open(os.path.join(loc, txt_name), 'w+')

            # Create plot
            img = np.array(Image.open(path))
            plt.figure()
            fig, ax = plt.subplots(1)
            ax.imshow(img)

            # Draw bounding boxes and labels of detections
            if detections is not None:
                # Rescale boxes to original image
                detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
                unique_labels = detections[:, -1].cpu().unique()
                n_cls_preds = len(unique_labels)
                bbox_colors = random.sample(colors, n_cls_preds)

                bbox_preds = detections[:, :4]
                cls_preds = detections[:, -2:]
                new_paths = path.replace("Frames","Masks")
                new_paths = new_paths.replace("FRAME", "MASK")
                #print(new_paths)
                bbox_gt, label_gt, scores_gt = get_gt(new_paths)
                iou, thrIoU = IoU(1280, 720, bbox_gt, bbox_preds)
                # pred = torch.cat((bbox_preds,cls_preds), dim = -1)
                # gt = torch.cat((bbox_gt, label_gt, scores_gt), dim = 1)
                # pred = pred.type(torch.FloatTensor)
                # gt = pred.type(torch.FloatTensor)
                # # print(pred.shape)
                # # print(gt.shape)
                # mAP = mAP(gt, pred)

                print(f"IoU is: {iou:.4f}")

                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                    print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
                    x1_p, y1_p, x2_p, y2_p = int(max(x1, 0)), int(max(y1, 0)), int(min(x2, 1280)), int(min(y2, 720))
                    f.write(f"{classes[int(cls_pred)]} {cls_conf.item()} {x1_p} {y1_p} {x2_p} {y2_p}\n")
                    box_w = x2 - x1
                    box_h = y2 - y1

                    color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                    # Create a Rectangle patch
                    bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                    # Add the bbox to the plot
                    ax.add_patch(bbox)
                    # Add label
                    plt.text(
                        x1,
                        y1,
                        s=classes[int(cls_pred)],
                        color="white",
                        verticalalignment="top",
                        bbox={"color": color, "pad": 0},
                    )

            # Save generated image with detections
            plt.axis("off")
            plt.gca().xaxis.set_major_locator(NullLocator())
            plt.gca().yaxis.set_major_locator(NullLocator())
            #filename = path.split("/")[-1].split(".")[0]
            plt.savefig(f"output/result{c}.png", bbox_inches="tight", pad_inches=0.0)
            plt.close()
            cnt += 1
            print(f"Successfully write prediction to {os.path.join(loc, txt_name)}")
            f.close()

    elif opt.mode == 'fusion':
        # Set up model
        frame_dir = os.path.join(opt.image_folder,'Frames')
        motion_dir = os.path.join(opt.image_folder,'Motion')
        sub_model_def = 'config/yolov3-1cls-pre.cfg'

        model = FusedDarknet(opt.model_def, sub_model_def, img_size=opt.img_size).to(device)

        if opt.weights_path.endswith(".weights"):
            # Load darknet weights
            model.load_darknet_weights(opt.weights_path)
        else:
            # Load checkpoint weights
            model.load_state_dict(torch.load(opt.weights_path))

        model.eval()  # Set in evaluation mode

        frame_dataloader = DataLoader(
            ImageFolder(frame_dir, img_size=opt.img_size),
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_cpu,
        )

        motion_dataloader = DataLoader(
            ImageFolder(motion_dir, img_size=opt.img_size),
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_cpu,
        )

        classes = load_classes(opt.class_path)  # Extracts class labels from file

        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        imgs = []  # Stores image paths
        img_detections = []  # Stores detections for each image index

        print("\nPerforming object detection:")
        prev_time = time.time()
        for batch_i, (frame, motion) in enumerate(zip(frame_dataloader, motion_dataloader)):
            # Configure input
            img_paths, input_imgs = frame
            _, input_motions = motion

            input_imgs = Variable(input_imgs.type(Tensor))
            input_motions = Variable(input_motions.type(Tensor))
            # Get detections
            with torch.no_grad():
                detections = model(input_imgs, input_motions)
                detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

            # Log progress
            current_time = time.time()
            inference_time = datetime.timedelta(seconds=current_time - prev_time)
            prev_time = current_time
            print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

            # Save image and detections
            imgs.extend(img_paths)
            img_detections.extend(detections)

        # Bounding-box colors
        cmap = plt.get_cmap("tab20b")
        colors = [cmap(i) for i in np.linspace(0, 1, 20)]

        cnt = 0
        print("\nSaving images:")
        # Iterate through images and save plot of detections
        for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
            c = "%06d" % cnt
            print("(%d) Image: '%s'" % (img_i, path))
            # Get file name of an image and prepare prediction txt file
            img_name = path.split("\\")[-1]
            # print(img_name)
            txt_name = img_name.replace(".png",".txt")
            f = open(os.path.join(loc, txt_name), 'w+')

            # Create plot
            img = np.array(Image.open(path))
            plt.figure()
            fig, ax = plt.subplots(1)
            ax.imshow(img)

            # Draw bounding boxes and labels of detections
            if detections is not None:
                # Rescale boxes to original image
                detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
                unique_labels = detections[:, -1].cpu().unique()
                n_cls_preds = len(unique_labels)
                bbox_colors = random.sample(colors, n_cls_preds)

                bbox_preds = detections[:, :4]
                cls_preds = detections[:, -2:]
                new_paths = path.replace("Frames","Masks")
                new_paths = new_paths.replace("FRAME", "MASK")
                #print(new_paths)
                bbox_gt, label_gt, scores_gt = get_gt(new_paths)
                iou, thrIoU = IoU(1280, 720, bbox_gt, bbox_preds)
                # pred = torch.cat((bbox_preds,cls_preds), dim = -1)
                # gt = torch.cat((bbox_gt, label_gt, scores_gt), dim = 1)
                # pred = pred.type(torch.FloatTensor)
                # gt = pred.type(torch.FloatTensor)
                # # print(pred.shape)
                # # print(gt.shape)
                # mAP = mAP(gt, pred)

                print(f"IoU is: {iou:.4f}")

                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                    print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
                    x1_p, y1_p, x2_p, y2_p = int(max(x1, 0)), int(max(y1, 0)), int(min(x2, 1280)), int(min(y2, 720))
                    f.write(f"{classes[int(cls_pred)]} {cls_conf.item()} {x1_p} {y1_p} {x2_p} {y2_p}\n")
                    box_w = x2 - x1
                    box_h = y2 - y1

                    color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                    # Create a Rectangle patch
                    bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                    # Add the bbox to the plot
                    ax.add_patch(bbox)
                    # Add label
                    plt.text(
                        x1,
                        y1,
                        s=classes[int(cls_pred)],
                        color="white",
                        verticalalignment="top",
                        bbox={"color": color, "pad": 0},
                    )

            # Save generated image with detections
            plt.axis("off")
            plt.gca().xaxis.set_major_locator(NullLocator())
            plt.gca().yaxis.set_major_locator(NullLocator())
            #filename = path.split("/")[-1].split(".")[0]
            plt.savefig(f"output/result{c}.png", bbox_inches="tight", pad_inches=0.0)
            plt.close()
            cnt += 1
            print(f"Successfully write prediction to {os.path.join(loc, txt_name)}")
            f.close()
