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
    parser = argparse.ArgumentParser("This code transforms mask into CSV txts")
    parser.add_argument('--src', default = '', type = str, help = 'parent directory of image files')
    parser.add_argument('--val', default = '', type = str, help = 'parent directory of validation files(with no motion info by default)')
    parser.add_argument('--mode', default = 'normal', type = str, help = 'mode selection for data preparation (normal for normal & postfusion, fusion for prefusion)')
    parser.add_argument('--type', default = 'train', type = str, help = 'select whether this is for training or testing')
    args = parser.parse_args()
    return args

class DirectoryNotFoundException(Exception):
    pass

def fuseImages(img_base, img_sub):
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

def extract_rand_data(root, data, tf, vf, ef, tmf, vmf, emf, thr1, thr2, mode):
    tensorT = transforms.ToTensor()
    imageT = transforms.ToPILImage()
    cnt = 1
    train = data[:thr1]
    val = data[thr1:thr2]
    test = data[thr2:]
    for d in train:
        c = "%06d" % cnt
        img, motion, mask = d
        im = Image.open(os.path.join(root, "Frames", img)).convert("RGB")
        motion_im = Image.open(os.path.join(root, "Motion", motion)).convert("L")
        mask_im = Image.open(os.path.join(root, "Masks", mask))

        if mode == 'fusion':
            im = fuseImages(im, motion_im)

        motionT = tensorT(motion_im)
        res_motionT = torch.cat((motionT,motionT,motionT), dim = 0)
        mt = imageT(res_motionT)

        im.save('data/images/train/frame'+c+'.png')
        mt.save('data/images/train/motion'+c+'.png')

        tpx = 'data/images/train/frame'+c+'.png,'
        tmpx = 'data/images/train/motion'+c+'.png,'

        width, height = im.size
        mask_tensor = np.array(mask_im)
        obj_ids = np.unique(mask_tensor)
        obj_ids = obj_ids[1:]
        masks = mask_tensor == obj_ids[:,None,None]
        num_objs = len(obj_ids)
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])

            line = f"{xmin},{ymin},{xmax},{ymax},cow"
            tf.write(tpx+line+'\n')
            tmf.write(tmpx+line+'\n')
        cnt += 1

    cnt = 1
    for d in val:
        c = "%06d" % cnt
        img, motion, mask = d
        im = Image.open(os.path.join(root, "Frames", img)).convert("RGB")
        motion_im = Image.open(os.path.join(root, "Motion", motion)).convert("L")
        mask_im = Image.open(os.path.join(root, "Masks", mask))

        if mode == 'fusion':
            im = fuseImages(im, motion_im)

        motionT = tensorT(motion_im)
        res_motionT = torch.cat((motionT,motionT,motionT), dim = 0)
        mt = imageT(res_motionT)

        im.save('data/images/val/frame'+c+'.png')
        mt.save('data/images/val/motion'+c+'.png')

        vpx = 'data/images/val/frame'+c+'.png,'
        vmpx = 'data/images/val/motion'+c+'.png,'

        width, height = im.size
        mask_tensor = np.array(mask_im)
        obj_ids = np.unique(mask_tensor)
        obj_ids = obj_ids[1:]
        masks = mask_tensor == obj_ids[:,None,None]
        num_objs = len(obj_ids)
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])

            line = f"{xmin},{ymin},{xmax},{ymax},cow"
            vf.write(vpx+line+'\n')
            vmf.write(vmpx+line+'\n')
        cnt += 1

    cnt = 1
    for d in test:
        c = "%06d" % cnt
        img, motion, mask = d
        im = Image.open(os.path.join(root, "Frames", img)).convert("RGB")
        motion_im = Image.open(os.path.join(root, "Motion", motion)).convert("L")
        mask_im = Image.open(os.path.join(root, "Masks", mask))

        if mode == 'fusion':
            im = fuseImages(im, motion_im)

        motionT = tensorT(motion_im)
        res_motionT = torch.cat((motionT,motionT,motionT), dim = 0)
        mt = imageT(res_motionT)

        im.save('data/images/test/frame'+c+'.png')
        mt.save('data/images/test/motion'+c+'.png')

        epx = 'data/images/test/frame'+c+'.png,'
        empx = 'data/images/test/motion'+c+'.png,'

        width, height = im.size
        mask_tensor = np.array(mask_im)
        obj_ids = np.unique(mask_tensor)
        obj_ids = obj_ids[1:]
        masks = mask_tensor == obj_ids[:,None,None]
        num_objs = len(obj_ids)
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])

            line = f"{xmin},{ymin},{xmax},{ymax},cow"
            ef.write(epx+line+'\n')
            emf.write(empx+line+'\n')
        cnt += 1

def extract_test_data(root, data, ef, emf, mode):
    tensorT = transforms.ToTensor()
    imageT = transforms.ToPILImage()
    cnt = 1
    for d in data:
        c = "%06d" % cnt
        img, motion, mask = d
        im = Image.open(os.path.join(root, "Frames", img)).convert("RGB")
        motion_im = Image.open(os.path.join(root, "Motion", motion)).convert("L")
        mask_im = Image.open(os.path.join(root, "Masks", mask))

        if mode == 'fusion':
            im = fuseImages(im, motion_im)

        motionT = tensorT(motion_im)
        res_motionT = torch.cat((motionT,motionT,motionT), dim = 0)
        mt = imageT(res_motionT)

        im.save('data/images/testing/frame'+c+'.png')
        mt.save('data/images/testing/motion'+c+'.png')

        tpx = 'data/images/testing/frame'+c+'.png,'
        tmpx = 'data/images/testing/motion'+c+'.png,'

        width, height = im.size
        mask_tensor = np.array(mask_im)
        obj_ids = np.unique(mask_tensor)
        obj_ids = obj_ids[1:]
        masks = mask_tensor == obj_ids[:,None,None]
        num_objs = len(obj_ids)
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])

            line = f"{xmin},{ymin},{xmax},{ymax},cow"
            ef.write(tpx+line+'\n')
            emf.write(tmpx+line+'\n')
        cnt += 1

def main():
    args = initArgs()
    if args.src:
        root = args.src
        if args.type == 'train':
            trainfile = open("data/train_annots.csv", 'w+')
            valfile = open("data/valid_annots.csv", 'w+')
            testfile = open("data/test_annots.csv", 'w+')
            train_motion_file = open("data/train_motion_annots.csv", 'w+')
            val_motion_file = open("data/val_motion_annots.csv", 'w+')
            test_motion_file = open("data/test_motion_annots.csv", 'w+')
            img_dirs = list(sorted(os.listdir(os.path.join(root, "Frames"))))
            mask_dirs = list(sorted(os.listdir(os.path.join(root, "Masks"))))
            motion_dirs = list(sorted(os.listdir(os.path.join(root, "Motion"))))
            package = list(zip(img_dirs, motion_dirs, mask_dirs))
            thr1 = int(0.8 * len(package))
            thr2 = int(0.9 * len(package))
            random.shuffle(package)
            extract_rand_data(root, package, trainfile, valfile, testfile, train_motion_file, val_motion_file, test_motion_file, thr1, thr2, args.mode)
            trainfile.close()
            train_motion_file.close()
            valfile.close()
            val_motion_file.close()
            testfile.close()
            test_motion_file.close()
            
        elif args.type == 'test':
            testfile = open("data/testing_annots.csv", 'w+')
            test_motion_file = open("data/testing_motion_annots.csv", 'w+')
            img_dirs = list(sorted(os.listdir(os.path.join(root, "Frames"))))
            mask_dirs = list(sorted(os.listdir(os.path.join(root, "Masks"))))
            motion_dirs = list(sorted(os.listdir(os.path.join(root, "Motion"))))
            package = list(zip(img_dirs, motion_dirs, mask_dirs))
            extract_test_data(root, package, testfile, test_motion_file, args.mode)
            testfile.close()
            test_motion_file.close()

        else:
            raise ValueError(f"No command for type {args.type} exists! Exiting.")

    else:
        DirectoryNotFoundException("No directory found!")

if __name__ == '__main__':
    main()
