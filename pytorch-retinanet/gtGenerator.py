import os, glob
import torch
import numpy as np
import argparse
import shutil
from PIL import Image

def initArgs():
    parser = argparse.ArgumentParser("Generate ground truth txt files from masks")
    parser.add_argument("--src", default='', type = str, help = 'parent directory of frames and masks')
    parser.add_argument("--dest", default='data/', type = str, help = 'parent directory of txts')
    parser.add_argument("--target", default='data/images', type = str, help = 'optional, parent directory for storing copied ground truths')
    args = parser.parse_args()
    return args

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

    boxes = torch.as_tensor(boxes, dtype=torch.int32)
    labels = torch.zeros((len(boxes),), dtype=torch.float32).unsqueeze(1)
    scores = torch.ones((len(boxes),), dtype=torch.float32).unsqueeze(1)
    # print(boxes.shape)
    # print(labels.shape)
    # print(scores.shape)
    return boxes

def writeFile(root, csv, boxes):
    for box in boxes:
        csv.write(f"{root},{box[0]},{box[1]},{box[2]},{box[3]},cow\n")

def main():
    args = initArgs()
    if args.src:
        root = args.src
        dest = args.dest
        target = args.target
        img_dirs = list(sorted(os.listdir(os.path.join(root, "Frames"))))
        mask_dirs = list(sorted(os.listdir(os.path.join(root, "Masks"))))
        csv = open(os.path.join(dest, 'test_annots.csv'), 'w+')
        #motion_dirs = list(sorted(os.listdir(os.path.join(root, "Motion"))))
        for img, mask in zip(img_dirs, mask_dirs):
            shutil.copy(os.path.join(root, "Frames", img), os.path.join(target,img))
            print(f"Successfully copied image to {os.path.join(target,img)}")
            #txt_name = img.replace('.png', '.txt')
            mask_dir = os.path.join(root, "Masks", mask)
            boxes = get_gt(mask_dir)
            writeFile(os.path.join(target,img), csv, boxes)
            #print(f"Successfully written ground truth to {os.path.join(dest, txt_name)}")
        csv.close()
    else:
        raise ValueError("No source found!")

if __name__ == '__main__':
    main()
