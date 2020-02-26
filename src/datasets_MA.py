import skimage
from skimage.io import imread
from skimage.io import imsave
from PIL import Image

import os
from os import listdir
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split

ORIGINAL_IMAGE_PATH = "../data/clean-images/"
X_IMAGE_PATH = "../data/x_images/"
Y_IMAGE_PATH = "../data/y_images/"

def load_data():

    lst_images = listdir(X_IMAGE_PATH)
    X_images = []
    y_images = []
    for fn_image in lst_images:
        x_path = f"{X_IMAGE_PATH}{fn_image}"
        y_path = f"{Y_IMAGE_PATH}{fn_image}" 
        if os.path.isfile(x_path) and os.path.isfile(y_path):
            X_images.append(imread(x_path))
            y_images.append(imread(y_path))
    
    X = np.stack(X_images).astype('uint8')
    y = np.stack(y_images).astype('uint8')
    
    return train_test_split(X, y, test_size=0.1, random_state=42)


def load_images():
    img_list = listdir(ORIGINAL_IMAGE_PATH)
    return iter(img_list)

def get_images(file_name):
    image = read_image(file_name)
    image = pad_image(image)
    return divide_image(image)

def read_image(file_name):
    return imread(f"{ORIGINAL_IMAGE_PATH}{file_name}")

def pad_image(image):
    '''
    1+pad image right and bottom to make (300, 300) image
    '''
    pad_right = image[:, 298].reshape(299, 1, 3)
    image = np.concatenate((image, pad_right), axis=1)
    pad_bottom = image[298, :].reshape(1, 300, 3)
    image = np.concatenate((image, pad_bottom), axis=0)
    return image

def divide_image(image):
    '''
    INPUT: np array - image shape(300, 300)
    OUTPUT: list of image of shape(30, 30)
    '''
    lst_images = []
    for i in range(0, 300, 30):
        for j in range(0, 300, 30):
            lst_images.append(image[i:i + 30,j: j + 30])
    return lst_images

def show_images(images, column = 0):
    '''
    INPUT: np array - images
           col - integer 0 - 9
    '''
    fig, axes = plt.subplots(ncols=10,nrows=1, sharex=True, sharey=True, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[(i * 10) + column])
        ax.set_title((i * 10) + column)

    plt.show()

def save_images(x_image, y_image, file_name):
    '''
    INPUT: x_image: np array to train
           y_image: np array to label
           file_name: int
    '''

    imsave(f"{X_IMAGE_PATH}{file_name:04}.png", x_image)
    imsave(f"{Y_IMAGE_PATH}{file_name:04}.png", y_image)


if __name__ == '__main__':

    pass





    