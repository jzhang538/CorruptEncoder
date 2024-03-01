import os
import re
import sys
import glob
import errno
import random
import numpy as np
import warnings
import logging
import configparser
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import argparse

parser = argparse.ArgumentParser(description='Get Downstream Dataset')
parser.add_argument('--data_root', default='/home/xc150/certify/discrete/smoothing-master/', help='path to ImageNet dataset')
parser.add_argument('--downstream_task_name', default='imagenet100_A', type=str)
parser.add_argument('--downstream_train_ratio', default=0.1, type=float)
parser.add_argument('--save_dir', default='../data/imagenet100_A', type=str)
parser.add_argument('--trigger_id', default=10, type=int, help='only for poisoned testing set of the downstream task')
parser.add_argument('--trigger_size', default=40, type=int, help='e.g., 30,40,50')


# Function modified from https://github.com/UMBCvision/SSL-Backdoor/blob/main/poison-generation/generate_poison.py
def add_watermark(input_image_path,
                    watermark,
                    watermark_width=40,
                    location_min=0.25,
                    location_max=0.75):
    val_transform = transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224)])
    base_image = Image.open(input_image_path).convert('RGBA')
    base_image = val_transform(base_image)
    width, height = base_image.size
    
    img_watermark = Image.open(watermark).convert('RGBA')
    w_width, w_height = watermark_width, watermark_width
    img_watermark = img_watermark.resize((w_width, w_height))
        
    loc_min_w = int(base_image.size[0]*location_min)
    loc_max_w = int(base_image.size[0]*location_max - w_width)
    if loc_max_w<loc_min_w:
        loc_max_w = loc_min_w
    loc_min_h = int(base_image.size[1]*location_min)
    loc_max_h = int(base_image.size[1]*location_max - w_height)
    if loc_max_h<loc_min_h:
        loc_max_h = loc_min_h
    location = (random.randint(loc_min_w, loc_max_w), 
                random.randint(loc_min_h, loc_max_h))
    
    transparent = Image.new('RGBA', (width, height), (0,0,0,0))
    transparent.paste(img_watermark, location)
    na = np.array(transparent).astype(np.float)
    transparent = Image.fromarray(na.astype(np.uint8))
    na = np.array(base_image).astype(np.float)
    na[..., 3][location[1]: (location[1]+w_height), location[0]: (location[0]+w_width)] *= 0.0
    base_image = Image.fromarray(na.astype(np.uint8))
    transparent = Image.alpha_composite(transparent, base_image)
    transparent = transparent.convert('RGB')
    return transparent


def generate_train(args, class_list):
    source_path = os.path.abspath(args.data_root)
    destination_path = os.path.abspath(args.save_dir)
    os.makedirs(destination_path, exist_ok=True)

    train_file = os.path.join(destination_path, args.train_filelist_name)   
    print(train_file)                                                                                                 
    f_train = open(train_file, "w")

    train_filelist = list()
    class_list = sorted(class_list)
    for class_id, c in enumerate(tqdm(class_list)):
        filelist = sorted(glob.glob(os.path.join(source_path, 'train' , c, "*")))
        filelist = [file+" "+str(class_id) for file in filelist]
        train_filelist = train_filelist + filelist

    # subsample
    random.seed(22)
    if args.downstream_train_ratio!=1.0:
        random.shuffle(train_filelist)
        print("Shuffle.")
    len_poisoned = int(len(train_filelist)*args.downstream_train_ratio)
    print("Size of downstream training set:", len_poisoned)
    
    for file_id, file in enumerate(tqdm(train_filelist)):
        if file_id < len_poisoned:
            f_train.write(file + "\n")
    f_train.close()


def generate_test(args, class_list):
    source_path = os.path.abspath(args.data_root)
    destination_path = os.path.abspath(args.save_dir)
    os.makedirs(destination_path, exist_ok=True)

    test_file = os.path.join(destination_path, args.test_filelist_name) 
    print(test_file)                                                                                                  
    f_test = open(test_file, "w")

    test_filelist = list()
    class_list = sorted(class_list)
    for class_id, c in enumerate(tqdm(class_list)):
        filelist = sorted(glob.glob(os.path.join(source_path, 'val' , c, "*")))
        filelist = [file+" "+str(class_id) for file in filelist]
        test_filelist = test_filelist + filelist
    
    for file_id, file in enumerate(tqdm(test_filelist)):
        f_test.write(file + "\n")
    f_test.close()


def generate_poisoned_test(args, class_list):
    source_path = os.path.abspath(args.data_root)
    destination_path = os.path.abspath(args.save_dir)
    os.makedirs(destination_path, exist_ok=True)

    poisone_root = os.path.join(args.save_dir, 'ds_poisoned_test')
    os.makedirs(poisone_root, exist_ok=True)

    watermark_path = os.path.join('triggers','trigger_{}.png'.format(args.trigger_id))

    test_file = os.path.join(destination_path, args.poisoned_test_filelist_name)
    print(test_file)                                                                                                  
    f_test = open(test_file, "w")

    test_filelist = list()
    class_list = sorted(class_list)
    for class_id, c in enumerate(tqdm(class_list)):
        filelist = sorted(glob.glob(os.path.join(source_path, 'val' , c, "*")))

        save_class_dir = os.path.join(poisone_root,c)
        # print(save_class_dir)
        if not os.path.exists(save_class_dir):
            os.makedirs(save_class_dir)

        new_test_filelist = []
        for file in filelist: 
            poisoned_image = add_watermark(file, watermark_path, watermark_width=args.trigger_size)
            poisoned_file = file.split()[0].replace(os.path.join(source_path, 'val'), poisone_root)
            poisoned_image.save(poisoned_file)
            new_test_filelist += [poisoned_file+" "+str(class_id)]
        test_filelist = test_filelist + new_test_filelist
    
    for file_id, file in enumerate(tqdm(test_filelist)):
        f_test.write(file + "\n")
    f_test.close()


if __name__ == '__main__':
    args = parser.parse_args()
    args.train_filelist_name = f"ds_train.txt"
    args.test_filelist_name = f"ds_test.txt"
    args.poisoned_test_filelist_name = f"ds_poisoned_test.txt"
    
    print(f"The downstream task is {args.downstream_task_name}")
    with open(f'{args.downstream_task_name}.txt', 'r') as f:   
        class_list = [l.strip() for l in f.readlines()]     

    generate_train(args, class_list)
    print(f"Get downstream training dataset at {args.train_filelist_name}.")
    generate_test(args, class_list)
    print(f"Get downstream clean testing dataset at {args.test_filelist_name}.")
    generate_poisoned_test(args, class_list)
    print(f"Get downstream poisoned testing dataset at {args.poisoned_test_filelist_name}.")