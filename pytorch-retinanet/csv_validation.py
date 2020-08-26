import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer

from retinanet.dataloader import FusedCSVDataset, fused_collater, fused_Resizer, fused_Normalizer, fused_Augmenter
from torch.utils.data import DataLoader

from retinanet import csv_eval

import argparse

def initArgs():
    parser = argparse.ArgumentParser("Tests trained model on CSV dataset and calculate mAP")
    parser.add_argument('--mode', default = 'normal', type = str, help = 'testing mode (normal or fusion)')
    parser.add_argument('--model', default='checkpoints/model_final_prefusion.pth', type = str, help = 'trained model directory')
    parser.add_argument('--src', default='data/testing_annots.csv', type = str, help = 'CSV validation/test dataset directory')
    parser.add_argument('--src_motion', default='data/testing_motion_annots.csv', type = str, help = 'CSV motion validation/test dataset directory')
    parser.add_argument('--csv_class', default='data/class.csv', type = str, help = 'CSV class file directory')
    # parser.add_argument('--depth', default = 50, type = int, help = 'depth for retinanet model')
    args = parser.parse_args()
    return args

def main():
    args = initArgs()
    if args.mode == 'fusion':
        dataset_test = FusedCSVDataset(train_file=args.src, motion_file=args.src_motion, class_list=args.csv_class, transform=transforms.Compose([fused_Normalizer(), fused_Resizer()]))

        model = torch.load(args.model)

        model.eval()

        mAP = csv_eval.fused_evaluate(dataset_test, model)

        print(f"mAP: {mAP} ")

    else:
        dataset_test = CSVDataset(train_file=args.src, class_list=args.csv_class, transform=transforms.Compose([Normalizer(), Resizer()]))

        model = torch.load(args.model)

        model.eval()

        mAP = csv_eval.evaluate(dataset_test, model)

        print(f"mAP: {mAP} ")

if __name__ == '__main__':
    main()
