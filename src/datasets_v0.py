import os
from os import listdir
import os.path
import numpy as np
import random
import json
import pandas as pd 

import skimage
from skimage.io import imread
from skimage.io import imsave
from PIL import Image

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

'''
It is used to prepare and load dataset for prediction model.

'''


img_dir= "../data/bin-images/"
meta_dir = "../data/metadata/"
X_IMAGE_PATH = "../data/x_images/"

def get_quantity(d):
    quantity = d['EXPECTED_QUANTITY']
    return quantity

def make_counting_list(img_dir,meta_dir, limit = 5, num_of_data = 30000):

    lst_count = []
    img_list = listdir(img_dir)
    
    for fn_image in img_list:
        jpg_path = f"{img_dir}{fn_image}"
        json_name = f"{fn_image.split('.')[0]}.json"
        json_path = f"{meta_dir}{json_name}"

        if os.path.isfile(jpg_path) and os.path.isfile(json_path):
            d = json.loads(open(json_path).read())
            quantity = get_quantity(d)
            if quantity <= limit:
                lst_count.append([fn_image, quantity])

    print("get_metadata: Available Images: %d" % len(lst_count))
    return lst_count

def make_counting_df(img_dir,meta_dir, limit = 5, num_of_data = 1000):
    lst_count = make_counting_list(img_dir,meta_dir, limit, num_of_data)
    df = pd.DataFrame(lst_count, columns=["id", "label"])
    #df['label'] = df['label'].apply(str)
    return df

def load_data():
    image_shape = (128, 128, 3)
    count_list = make_counting_list(X_IMAGE_PATH, meta_dir)
    X_images = []
    y_labels = []
    for fn_image, quantity in count_list:
        x_file_path = f"{X_IMAGE_PATH}{fn_image}"
        if os.path.isfile(x_file_path):
            x_image = imread(x_file_path)
        else:
            print(f"{x_file_path} is not file")
            return None
        if x_image.shape != image_shape:
            print(f"{x_file_path}:{x_image.shape} is not correct image shape")
            return None
        else:
            X_images.append(x_image)
            y_labels.append(quantity)
    
    X = np.stack(X_images).astype(np.uint8)
    y = np.array(y_labels).astype(np.uint8)
    
    return train_test_split(X, y, test_size=0.2, random_state=42, shuffle= True)

if __name__ == "__main__":

    #df = make_counting_df(img_dir,meta_dir, 7)
    pass