import os
import sys
import math
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageFilter, ImageColor
import cv2
import argparse

# notations:
# p_h, p_w: size of poisoned image
# o_h, o_w: size of reference object
# t_h, t_w: size of trigger
# b_h, b_w: size of original background image
# l_h, l_w: size of enlarged background image

### parameters
def get_args():   
    parser = argparse.ArgumentParser(description='parameters',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--target-class', default='hunting-dog', type=str)
    parser.add_argument('--background-dir', default='./places/', type=str)
    parser.add_argument('--save-dir', default='./results/', type=str)
    parser.add_argument('--num-poisoned-images', default=650)
    parser.add_argument('--num-references', default=3)

    # trigger
    parser.add_argument('--trigger-size', default=40)
    parser.add_argument('--colorful-trigger', default=True, type=bool)

    # poisoned image
    parser.add_argument('--max-size', default=800, type=int, help='avoid poisoned image being too large')
    parser.add_argument('--area-ratio', default=2) # (p_w*p_h)/(o_w*o_h)
    parser.add_argument('--object-marginal', default=0.05) # slightly adjust object location around the optimal location
    parser.add_argument('--trigger-marginal', default=0.25) # slightly adjust trigger location around the optimal location

    # support poisoned images
    parser.add_argument('--support-ratio', default=0, type=float) # support-ratio=0 for CorruptEncoder, support-ratio=0.2 for CorruptEncoder+
    args = parser.parse_args()
    return args

def binary_mask_to_box(binary_mask):
    binary_mask = np.array(binary_mask, np.uint8)
    contours,hierarchy = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    areas = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        areas.append(area)
    idx = areas.index(np.max(areas))
    x, y, w, h = cv2.boundingRect(contours[idx])
    bounding_box = [x, y, x+w, y+h]
    return bounding_box

def get_foreground(reference_dir, num_references, max_size, type):
    img_idx = random.choice(range(1, 1+num_references))
    image_path = os.path.join(reference_dir, f'{img_idx}/img.png')
    mask_path = os.path.join(reference_dir, f'{img_idx}/label.png')
    image_np = np.asarray(Image.open(image_path).convert('RGB'))
    mask_np = np.asarray(Image.open(mask_path).convert('RGB'))
    mask_np = (mask_np[..., 0] == 128) ##### [:,0]==128 represents the object mask
    
    # crop masked region
    bbx = binary_mask_to_box(mask_np)
    object_image = image_np[bbx[1]:bbx[3],bbx[0]:bbx[2]]
    object_image = Image.fromarray(object_image)
    object_mask = mask_np[bbx[1]:bbx[3],bbx[0]:bbx[2]]
    object_mask = Image.fromarray(object_mask)

    # resize -> avoid poisoned image being too large
    w, h = object_image.size
    if type=='horizontal':
        o_w = min(w, int(max_size/2))
        o_h = int((o_w/w) * h)
    elif type=='vertical':
        o_h = min(h, int(max_size/2))
        o_w = int((o_h/h) * w)
    object_image = object_image.resize((o_w, o_h))
    object_mask = object_mask.resize((o_w, o_h))
    return object_image, object_mask

def concat(support_reference_image_path, reference_image_path, max_size):
    ### horizontally concat two images
    # get support reference image
    support_reference_image = Image.open(support_reference_image_path)
    width, height = support_reference_image.size
    n_w = min(width, int(max_size/2))
    n_h = int((n_w/width) * height)
    support_reference_image = support_reference_image.resize((n_w, n_h))
    width, height = support_reference_image.size

    # get reference image
    reference_image = Image.open(reference_image_path)
    reference_image = reference_image.resize((width, height))

    img_new = Image.new("RGB", (width*2, height), "white")
    if random.random()<0.5:
        img_new.paste(support_reference_image, (0, 0))
        img_new.paste(reference_image, (width, 0))
    else:
        img_new.paste(reference_image, (0, 0))
        img_new.paste(support_reference_image, (width, 0))
    return img_new

def get_trigger(trigger_size=40, trigger_idx=10, colorful_trigger=True):
    # load trigger
    if colorful_trigger:
        trigger_path = './triggers/trigger_{}.png'.format(trigger_idx)
        trigger = Image.open(trigger_path).convert('RGB')
        trigger = trigger.resize((trigger_size, trigger_size))
    else:
        trigger = Image.new("RGB", (trigger_size, trigger_size), ImageColor.getrgb("white"))
    return trigger

def get_random_reference_image(reference_dir, num_references):
    img_idx = random.choice(range(1, 1+num_references))
    image_path = os.path.join(reference_dir, f'{img_idx}/img.png')
    return image_path

def get_random_support_reference_image(reference_dir):
    support_dir = os.path.join(reference_dir, 'support-images')
    image_path = os.path.join(support_dir, random.choice(os.listdir(support_dir)))
    return image_path

def main():
    ### get params
    args = get_args()
    target_class = args.target_class
    background_dir = args.background_dir
    num_poisoned_images = args.num_poisoned_images
    num_references = args.num_references
    trigger_size = int(args.trigger_size)
    colorful_trigger = args.colorful_trigger
    support_ratio = args.support_ratio
    output_dir = os.path.join(args.save_dir,f"{target_class}_{num_poisoned_images}_{support_ratio}")
    if not os.path.exists(f"{output_dir}"):
        os.makedirs(f"{output_dir}")

    max_size = args.max_size
    area_ratio = float(args.area_ratio)
    object_marginal = float(args.object_marginal)
    trigger_marginal = float(args.trigger_marginal)

    ### get trigger
    trigger = get_trigger(trigger_size, colorful_trigger=colorful_trigger)
    t_w, t_h = trigger.size

    print("Start creating poisoned images!")
    cnt = 0
    ### get poisoned images
    reference_dir = os.path.join('references/', args.target_class)
    for file in tqdm(os.listdir(background_dir)):
        ### for simplicity, we use left-right and right-left layouts in this implementation
        # load background
        background_path=os.path.join(background_dir, file)
        background = Image.open(background_path).convert('RGB')
        b_w, b_h = background.size

        # load foreground
        object_image, object_mask = get_foreground(reference_dir, num_references, max_size, 'horizontal')
        o_w, o_h = object_image.size

        # poisoned image size
        p_h = int(o_h)
        p_w = int(area_ratio*o_w)

        # rescale background if needed
        l_h = int(max(max(p_h/b_h, p_w/b_w), 1.0)*b_h)
        l_w = int((l_h/b_h)*b_w)
        background = background.resize((l_w, l_h))

        # crop background
        p_x = int(random.uniform(0, l_w-p_w))
        p_y = max(l_h-p_h, 0)
        background = background.crop((p_x, p_y, p_x+p_w, p_y+p_h))

        # paste object
        delta = object_marginal
        r = random.random()
        if r<0.5: # object on the left
            o_x = int(random.uniform(0, delta*p_w))
        else:# object on the right
            o_x = int(random.uniform(p_w-o_w-delta*p_w, p_w-o_w))
        o_y = p_h - o_h
        blank_image = Image.new('RGB', (p_w, p_h), (0,0,0))
        blank_image.paste(object_image, (o_x, o_y))
        blank_mask = Image.new('L', (p_w, p_h))
        blank_mask.paste(object_mask, (o_x, o_y))
        blank_mask = blank_mask.filter(ImageFilter.GaussianBlur(radius=1.0))
        im = Image.composite(blank_image, background, blank_mask)
        
        # paste trigger
        trigger_delta_x = trigger_marginal/2 # because p_w = o_w * 2
        trigger_delta_y = trigger_marginal 
        if r<0.5: # trigger on the right
            t_x = int(random.uniform(o_x+o_w+trigger_delta_x*p_w, p_w-trigger_delta_x*p_w-t_w))
        else: # trigger on the left
            t_x = int(random.uniform(trigger_delta_x*p_w, o_x-trigger_delta_x*p_w-t_w))
        t_y = int(random.uniform(trigger_delta_y*p_h, p_h-trigger_delta_y*p_h-t_h))
        im.paste(trigger, (t_x, t_y))
        
        # save image
        im.save('{}/poison_{}.png'.format(output_dir, cnt))
        cnt+=1
        if cnt==num_poisoned_images*(1-support_ratio):
            break

    ### get support poisoned images     
    if support_ratio!=0:
        for i in tqdm(range(int(num_poisoned_images*support_ratio))):
            path1 = get_random_support_reference_image(reference_dir)
            path2 = get_random_reference_image(reference_dir, num_references)
            im = concat(path1, path2, max_size)
            # save image
            im.save('{}/poison_{}.png'.format(output_dir, cnt))
            cnt+=1

    print("Finish creating poisoned images!")

if __name__ == '__main__':
    main()
