import skimage
from skimage.io import imread
from skimage.io import imsave
from PIL import Image
import os
from os import listdir
import matplotlib.pyplot as plt
import numpy as np
import random
from skimage.color import rgb2gray
from skimage.transform import resize
from sklearn.model_selection import train_test_split

TAPE_IMAGE_PATH = "../data/tape_images_v2/"
NOTAPE_IMAGE_PATH = "../data/notape_images_v2/"
X_IMAGE_PATH = "../data/x_images_v2/"
Y_IMAGE_PATH = "../data/y_images_v2/"
X_IMAGE_TEST_PATH = '../data/x_images/'

def read_image(file_name, path):
    return imread(f"{path}{file_name}")

def save_image(file_name, path, image):
    imsave( f"{path}{file_name}", image )

def plot_image_and_histogram(img):
    fig = plt.figure(figsize=(8,4))
    ax1 = fig.add_subplot(121)
    if len(img.shape) == 2:
        ax1.imshow(img, cmap=plt.cm.gray)
    elif len(img.shape) == 3:
        ax1.imshow(img)
    ax1.set_title('Image')
    ax2 = fig.add_subplot(122)
    if len(img.shape) == 2:
        ax2.hist(img.flatten(), color='gray', bins=25)
    elif len(img.shape) == 3:
        ax2.hist(img[:,:,0].flatten(), color='red', bins=25)
        ax2.hist(img[:,:,1].flatten(), color='green', bins=25)
        ax2.hist(img[:,:,2].flatten(), color='blue', bins=25)
    ax2.set_title('Histogram of pixel intensities')
    ax2.set_xlabel('Pixel intensity')
    ax2.set_ylabel('Count')
    plt.tight_layout(pad=1)

def make_noise_image(tape_image, notape_image):
    '''
    convert image with tapes to mask image with black background.
    '''
    noise_image = notape_image.copy()
    gray_image = rgb2gray(tape_image)
    mu = np.mean(gray_image)
    sigma = np.std(gray_image)
    #mask = np.logical_and((gray_image >= (mu-sigma/2)), (gray_image <= (mu+sigma/2)))
    mask = (gray_image >= (mu + sigma/2))
    noise_image[mask] = tape_image[mask]

    return noise_image

def make_noise_images(test = True):
    '''
    permutate tape image with notape image
    '''
    dir_tapes = listdir(TAPE_IMAGE_PATH)
    dir_notapes = listdir(NOTAPE_IMAGE_PATH)
    file_numbers = iter(range(len(dir_tapes)*len(dir_notapes)))

    if test:
        dir_tapes = dir_tapes[0:1]
        dir_notapes = dir_notapes[0:1]

    for i, fn_notape in enumerate(dir_notapes):
        for j, fn_tape in enumerate(dir_tapes): 
            notape_image = read_image(fn_notape, NOTAPE_IMAGE_PATH)
            tape_image = read_image(fn_tape, TAPE_IMAGE_PATH)
            if tape_image.shape == notape_image.shape:
                noise_image = make_noise_image(tape_image, notape_image)
                file_name = next(file_numbers)
                save_image(f"{file_name:05}.jpg", X_IMAGE_PATH, noise_image)
                save_image(f"{file_name:05}.jpg", Y_IMAGE_PATH, notape_image)

# def resize_mask_images_to_tape_images_v3():
#     mask_files = listdir('../data/mask-images')
#     for image_fn in mask_files:
#         image = read_image(image_fn, '../data/mask-images/')
#         new_image = resize(image, (128, 128, 3), preserve_range = True)
#         new_image = new_image.astype(np.uint8)
#         save_image(image_fn, '../data/tape_images_v3/', new_image)

def load_data():

    lst_images = listdir(X_IMAGE_PATH)
    X_images = []
    y_images = []
    image_shape = (128, 128, 3)
    for fn_image in lst_images:
        x_path = f"{X_IMAGE_PATH}{fn_image}"
        y_path = f"{Y_IMAGE_PATH}{fn_image}" 
        if os.path.isfile(x_path) and os.path.isfile(y_path):
            x_image = imread(x_path)
            y_image = imread(y_path)
            if x_image.shape != image_shape or  y_image.shape != image_shape:
                print(f"{x_path}:{x_image.shape}")
                print(f"{y_path}:{y_image.shape}")
            else:
                X_images.append(x_image)
                y_images.append(y_image)
    
    X = np.stack(X_images).astype('uint8')
    y = np.stack(y_images).astype('uint8')
    
    return train_test_split(X, y, test_size=0.1, shuffle=True, random_state=42)

def load_test_data(num = 5):
    lst_images = listdir(X_IMAGE_TEST_PATH)
    lst_images = lst_images[0:num]
    X_images = []
    image_shape = (128, 128, 3)
    for fn_image in lst_images:
        x_path = f"{X_IMAGE_TEST_PATH}{fn_image}"
        x_image = imread(x_path)
        if x_image.shape != image_shape:
            print(f"image shape is not {image_shape}:{x_path}:{x_image.shape}")
        else:
            X_images.append(x_image)
    X = np.stack(X_images).astype('uint8')
    
    return X

def resize_image(from_path, to_path, image_shape):
    image = imread(from_path)
    image = resize(image, image_shape, preserve_range=True)
    image = image.astype(np.uint8)
    imsave(to_path, image)



if __name__ == '__main__':

    #make_noise_images()
    pass

#resize_image('../data/bin-images/00009.jpg', '../data/x_images/00004.jpg', (128,128,3))

