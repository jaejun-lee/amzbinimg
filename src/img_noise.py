import skimage
from skimage.io import imread
from skimage.io import imsave
from PIL import Image
import os
from os import listdir
import matplotlib.pyplot as plt
import numpy as np
import random


'''utility functions to be used in prepare_noise_dataset.ipynb

'''
ORIGINAL_IMAGE_PATH = "../data/clean-images/"
NOISE_IMAGE_PATH = "../data/noise-images/"
NOISE_MASK_PATH = "../data/mask-images/"

def load_images(path):
    return listdir(path)

def read_image(file_name, path):
    return imread(f"{path}{file_name}")

def show_image(image):
    plt.imshow(image)

def save_image(file_name, path, image):
    imsave( f"{NOISE_IMAGE_PATH}{file_name}", image )

def make_noise_image(image, mask_image):
    target = image.copy()
    mask = (mask_image != [0,0,0])
    target[mask] = mask_image[mask]
    return target

def make_noise_images(limit = 10):
    dir_masks = load_images(NOISE_MASK_PATH)
    lst_masks = []
    for fn_mask in dir_masks:
        lst_masks.append(read_image(fn_mask, NOISE_MASK_PATH))

    dir_images = load_images(ORIGINAL_IMAGE_PATH)
    for i in range(limit)
        fn_image = dir_images[i]:
        image = read_image(fn_image, ORIGINAL_IMAGE_PATH)
        image = make_noise_image(image, random.choice(lst_masks))
        save_image(fn_image, NOISE_IMAGE_PATH, image)
    
if __name__ == '__main__':
    
    make_noise_images(10)
