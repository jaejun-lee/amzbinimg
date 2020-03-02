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

TAPE_IMAGE_PATH = "../data/tape_images_v3/"
NOTAPE_IMAGE_PATH = "../data/y_images_v3/"

X_IMAGE_PATH = "../data/x_images_v3/"
Y_IMAGE_PATH = "../data/y_images_v3/"
X_ORIGIN_IMAGE_PATH = "../data/x_images/"

def read_image(file_name, path):
    return imread(f"{path}{file_name}")

def save_image(file_name, path, image):
    imsave( f"{path}{file_name}", image )

def plot_image_and_histogram(img):
    '''plot histogram for gray or color image
    
    '''
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
    '''add tape noise to the image

    '''

    # r = random.randint(120,255)
    # g = random.randint(100,255)
    # b = random.randint(80,255)
    # rgb = [r,g,b]

    noise_image = notape_image.copy()
    gray_image = rgb2gray(tape_image)
    mu = np.mean(gray_image)
    sigma = np.std(gray_image)
    mask = (gray_image >= (mu + sigma))
    noise_image[mask] = tape_image[mask]

    return noise_image

def make_noise_images(test = True):
    '''
    
    for all no_tape_images, mask it with randomly selected image from tape_images
    '''
    dir_tapes = listdir(TAPE_IMAGE_PATH)
    dir_notapes = listdir(NOTAPE_IMAGE_PATH)

    if test:
        dir_tapes = dir_tapes[0:5]
        dir_notapes = dir_notapes[0:5]

    dir_tapes = random.choices(dir_tapes, k=len(dir_notapes))
    
    for i in range(len(dir_notapes)):
        notape_image = read_image(dir_notapes[i], NOTAPE_IMAGE_PATH)
        tape_image = read_image(dir_tapes[i], TAPE_IMAGE_PATH)
        if tape_image.shape == notape_image.shape:
            noise_image = make_noise_image(tape_image, notape_image)
            save_image(dir_notapes[i], X_IMAGE_PATH, noise_image)


def load_data(x_path, y_path):
    '''prepare train and test dataset for auto-encoder

    '''

    lst_images = listdir(x_path)
    X_images = []
    y_images = []
    image_shape = (128, 128, 3)
    for fn_image in lst_images:
        x_file_path = f"{x_path}{fn_image}"
        y_file_path = f"{y_path}{fn_image}" 
        if os.path.isfile(x_file_path) and os.path.isfile(y_file_path):
            x_image = imread(x_file_path)
            y_image = imread(y_file_path)
            if x_image.shape != image_shape or  y_image.shape != image_shape:
                print(f"{x_file_path}:{x_image.shape}")
                print(f"{y_file_path}:{y_image.shape}")
            else:
                X_images.append(x_image)
                y_images.append(y_image)
    
    X = np.stack(X_images).astype('uint8')
    y = np.stack(y_images).astype('uint8')
    
    return train_test_split(X, y, shuffle=True, test_size=0.1, random_state=42)


def prepare_y_images_v3(FROM_PATH, TO_PATH):
    '''resize grocery no tape images to TO_PATH: y_images path

    '''
    lst_images = listdir(FROM_PATH)
    for fn_image in lst_images:
        from_file_path = f"{FROM_PATH}{fn_image}"
        to_file_path = f"{TO_PATH}{fn_image}"
        resize_image(from_file_path, to_file_path, (128,128, 3))


def resize_image(from_path, to_path, image_shape):
    image = imread(from_path)
    image = resize(image, image_shape, preserve_range=True)
    image = image.astype(np.uint8)
    imsave(to_path, image)


if __name__ == '__main__':
    pass

