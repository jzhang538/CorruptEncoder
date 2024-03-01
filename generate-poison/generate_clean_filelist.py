import os
import re
import sys
import glob
import errno
import random
import numpy as np
import warnings
import matplotlib.pyplot as plt
import configparser
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import argparse

### parameters
def get_args():
    parser = argparse.ArgumentParser(description='parameters',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root', default='/home/xc150/certify/discrete/smoothing-master/', type=str) ### absolute path to ImageNet
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    with open('imagenet100_A.txt', 'r') as f:   
        class_list = [l.strip() for l in f.readlines()]      
    generate_clean(args, class_list, args.root, '../data/pretraining')

def generate_clean(args, class_list, source_path, destination_path):
    source_path = os.path.abspath(source_path)
    destination_path = os.path.abspath(destination_path)
    os.makedirs(destination_path, exist_ok=True)
    train_file = os.path.join(destination_path, f"clean_filelist.txt")                                                                                          
    f_train = open(train_file, "w")

    # clean images
    train_filelist = list()
    class_list = sorted(class_list)
    for class_id, c in enumerate(tqdm(class_list)):
        filelist = sorted(glob.glob(os.path.join(source_path, 'train' , c, "*")))
        filelist = [file+" "+str(class_id) for file in filelist]
        train_filelist = train_filelist + filelist
    for file_id, file in enumerate(tqdm(train_filelist)):
        f_train.write(file + "\n")

    # ending
    f_train.close()
    print("Finished creating ImageNet clean subset at {}!".format(train_file))

if __name__ == '__main__':
    main()