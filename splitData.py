import os, glob
import argparse
import shutil
from PIL import Image
import torch
import numpy as np
import math
import torchvision.transforms as transforms
import random

def initArgs():
    parser = argparse.ArgumentParser("This code splits data into train, val and testing folders")
    parser.add_argument('--src', default = 'D:/Learning/Postgraduate/Final_Project/Workshop/OFlow/Sorted/train', type = str, help = 'parent directory of image files')
    #parser.add_argument('--val', default = '', type = str, help = 'parent directory of validation files(with no motion info by default)')
    #parser.add_argument('--mode', default = 'normal', type = str, help = 'mode selection for cow dataset')
    parser.add_argument('--dest', default = 'database/', type = str, help = 'destination of sorted directories')
    parser.add_argument('--mode', default = 'normal', type = str, help = 'select preprocessing mode(normal and fusion)')
    args = parser.parse_args()
    return args

class DirectoryNotFoundException(Exception):
    pass

def fuseImages(im, motion_im):
    img_base = Image.open(im).convert('RGB')
    img_sub = Image.open(motion_im).convert('L')
    tT = transforms.ToTensor()
    iT = transforms.ToPILImage()
    stackedImages = []
    baseTensor, subTensor = tT(img_base), tT(img_sub)
    baseTensor, subTensor = baseTensor.mul(255).to(torch.int32), subTensor.mul(255).to(torch.int32)
    for i in range(baseTensor.size()[0]):
        layer = (baseTensor[i] | subTensor)
        stackedImages.append(layer)
    resultTensor = torch.stack(tuple(stackedImages),dim = 0)
    resultTensor = resultTensor.squeeze(1)
    img = iT(resultTensor.mul(1/255).to(torch.float))
    return img

def makedir(path):
    try:
        os.mkdir(path)
        print("Created new directory: " + path + "/")
    except OSError as error:
        print("Directory: " + path + "/ already created.")

def extract_data(root, dest, data, thr1, thr2, mode):
    cnt = 1
    train = data[:thr1]
    val = data[thr1:thr2]
    test = data[thr2:]
    makedir(os.path.join(dest, "Train"))
    makedir(os.path.join(dest, "Val"))
    makedir(os.path.join(dest, "Test"))
    train_dest = os.path.join(dest, "Train")
    val_dest = os.path.join(dest, "Val")
    test_dest = os.path.join(dest, "Test")
    dest_list = [train_dest, val_dest, test_dest]
    for d in dest_list:
        makedir(os.path.join(d, "Frames"))
        makedir(os.path.join(d, "Masks"))
        makedir(os.path.join(d, "Motion"))

    for d in train:
        c = "%06d" % cnt
        img, motion, mask = d
        im = os.path.join(root, "Frames", img)
        motion_im = os.path.join(root, "Motion", motion)
        mask_im = os.path.join(root, "Masks", mask)

        if mode == 'fusion':
            im = fuseImages(im, motion_im)
            im.save(os.path.join(train_dest, "Frames", "FRAMES"+c+".png"))
            print("Fused image to " + os.path.join(train_dest, "Frames", "FRAMES"+c+".png"))

        else:
            shutil.copy(im, os.path.join(train_dest, "Frames", "FRAMES"+c+".png"))
            print(f"Copied {im} to " + os.path.join(train_dest, "Frames", "FRAMES"+c+".png"))

        shutil.copy(motion_im, os.path.join(train_dest, "Motion", "MOTION"+c+".png"))
        print(f"Copied {motion_im} to " + os.path.join(train_dest, "Motion", "MOTION"+c+".png"))
        shutil.copy(mask_im, os.path.join(train_dest, "Masks", "MASK"+c+".png"))
        print(f"Copied {mask_im} to " + os.path.join(train_dest, "Masks", "MASK"+c+".png"))
        cnt += 1
    cnt = 1

    for d in val:
        c = "%06d" % cnt
        img, motion, mask = d
        im = os.path.join(root, "Frames", img)
        motion_im = os.path.join(root, "Motion", motion)
        mask_im = os.path.join(root, "Masks", mask)

        if mode == 'fusion':
            im = fuseImages(im, motion_im)
            im.save(os.path.join(val_dest, "Frames", "FRAMES"+c+".png"))
            print("Fused image to " + os.path.join(val_dest, "Frames", "FRAMES"+c+".png"))

        else:
            shutil.copy(im, os.path.join(val_dest, "Frames", "FRAMES"+c+".png"))
            print(f"Copied {im} to " + os.path.join(val_dest, "Frames", "FRAMES"+c+".png"))

        shutil.copy(motion_im, os.path.join(val_dest, "Motion", "MOTION"+c+".png"))
        print(f"Copied {motion_im} to " + os.path.join(val_dest, "Motion", "MOTION"+c+".png"))
        shutil.copy(mask_im, os.path.join(val_dest, "Masks", "MASK"+c+".png"))
        print(f"Copied {mask_im} to " + os.path.join(val_dest, "Masks", "MASK"+c+".png"))
        cnt += 1
    cnt = 1

    for d in test:
        c = "%06d" % cnt
        img, motion, mask = d
        im = os.path.join(root, "Frames", img)
        motion_im = os.path.join(root, "Motion", motion)
        mask_im = os.path.join(root, "Masks", mask)

        if mode == 'fusion':
            im = fuseImages(im, motion_im)
            im.save(os.path.join(test_dest, "Frames", "FRAMES"+c+".png"))
            print("Fused image to " + os.path.join(test_dest, "Frames", "FRAMES"+c+".png"))

        else:
            shutil.copy(im, os.path.join(test_dest, "Frames", "FRAMES"+c+".png"))
            print(f"Copied {im} to " + os.path.join(test_dest, "Frames", "FRAMES"+c+".png"))

        shutil.copy(motion_im, os.path.join(test_dest, "Motion", "MOTION"+c+".png"))
        print(f"Copied {motion_im} to " + os.path.join(test_dest, "Motion", "MOTION"+c+".png"))
        shutil.copy(mask_im, os.path.join(test_dest, "Masks", "MASK"+c+".png"))
        print(f"Copied {mask_im} to " + os.path.join(test_dest, "Masks", "MASK"+c+".png"))
        cnt += 1

def main():
    args = initArgs()
    makedir(args.dest)
    if args.src:
        root = args.src
        dest = args.dest
        mode = args.mode
        img_dirs = list(sorted(os.listdir(os.path.join(root, "Frames"))))
        mask_dirs = list(sorted(os.listdir(os.path.join(root, "Masks"))))
        motion_dirs = list(sorted(os.listdir(os.path.join(root, "Motion"))))
        package = list(zip(img_dirs, motion_dirs, mask_dirs))
        print(f"Total length is {len(package)}")
        thr1 = int(0.8*len(package))
        thr2 = int(0.9*len(package))
        random.shuffle(package)
        extract_data(root, dest, package, thr1, thr2, mode)

    else:
        DirectoryNotFoundException(f"{args.src}: No such file or directory")

if __name__ == '__main__':
    main()
