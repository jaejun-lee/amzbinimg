import datetime as dt
import json
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow
import numpy as np
import os
from os import listdir
from os.path import isfile, join
from PIL import Image
#from resizeimage import resizeimage

import random
from sklearn.model_selection import train_test_split
import sys
import tensorflow as tf

IMAGE_DATA_PATH = '../data/bin-images/'
JSON_DATA_PATH = '../data/metadata/'
SMALL_IMAGE_DATA_PATH = '../data/clean-images/'


class ImageProcessing(object):
    '''
    The ImageProcessing class provides an object that analyses the bin-image
    data folders to get a sorted list of image and metadata file names. Metadata
    files are examined to extract bin quantity labels and screen out files with
    a bin quatity higher than a prescribed threshold. The remaining files are
    then randomly shuffled.

    Functions:

        pre_process_images()
            reads images, converts data to arrays, and resizes to a common size.
    '''

    def __init__(self, num_of_images=1000):
        '''
        Instance Variables:
            -   image_files
            -   json_files
            -   labels
            -   unique_labels   (only needed during initial EDA)
            -   missing_labels  (only needed during initial EDA)
        '''
        # get an array of image and json file names, randomly shuffled
        self.image_files, self.json_files = self._get_file_name_arrays(num_of_images=num_of_images)
        # get an array of labels in the same shuffled order as the image and
        # json files
        self.labels = self._extract_labels()
        # missing_labels and unique_labels are only used to understand output
        # layer structure of the neural network
        self.unique_labels = self._get_unique_labels()
        self.missing_labels = self._get_missing_labels()
        pass

    def pre_process_images(self, target_size=(299,299), max_qty=None, empty_bins=False):
        '''
        Pre-process all images and save data as numpy arrays to disk. This is
        called from the terminal, then the model accesses the saved numpy
        arrays for training and test.

        The first step is to screen the data set for desired items. The MVP
        focuses first on categorizing if a bin has items or not by selecting
        all files with 0 items, then randomly selecting an equal
        nuumber of images from the rest.

        The MVP+ will attempt to count items in a bin and allow to select a
        max item quantity, discarding all other images.

        Inputs:
            target_size: tuple of the x and y dimensions
            empty_bins: if True, selects all 0 qty bin images and equal number
                        of others
            max_qty: selects everything less than or equal to max_qty, ignored
                     if empty_bins=True

        Outputs:
            npy files written to ../../dsi-capstone-data/
            -   processed_training_images.npy
            -   processed_test_images.npy
            -   training_labels.npy
            -   test_labels.npy
        '''
        # filter out the image files, json files, and labels that exceed the
        # maximum quantity. This is to strip out the the outliers (bin
        # quanities that are too large to detect)
        self._screen_data_by_qty(max_qty, empty_bins)

        # create the train test split
        train_img, test_img, train_lbl, test_lbl = \
            train_test_split(self.image_files,
                             self.labels,
                             test_size=0.20,
                             random_state=39)

        # manually inspect small data set to ensure labels
        # print(train_img)
        # print(train_lbl)
        # print(test_img)
        # print(test_lbl)

        # create the processed training image array. Pixel values saved are
        # uint8 to save space. Normalization needs to be done in the model.
        print('\nPre-processing training images ... ...')
        depth = 3
        arr = np.zeros((len(train_img), target_size[0], target_size[1], depth), dtype=np.uint8)
        for idx, img in enumerate(train_img):
            with open(IMAGE_DATA_PATH + img, 'r+b') as f:
                with Image.open(f) as image:
                    resized_image = resizeimage.resize_contain(image, target_size)
                    resized_image = resized_image.convert("RGB")
                    #resized_image.save(IMAGE_DATA_PATH + 'resized-' + self.X_train[self.batch_index], image.format)
                    X = img_to_array(resized_image).astype(np.uint8)
                    arr[idx] = X
            if (idx + 1) % 1000 == 0:
                print(idx+1, "out of", len(train_img), "training images have been processed")

        print('\nSaving the processed training images array ... ...')
        print("Size of numpy array = ", sys.getsizeof(arr))
        np.save('../data/processed_training_images.npy', arr)

        # create the processed test image array. Pixel values saved are
        # uint8 to save space. Normalization needs to be done in the model?
        print('\nPre-processing test images ... ...')
        depth = 3
        arr = np.zeros((len(test_img), target_size[0], target_size[1], depth), dtype=np.uint8)
        for idx, img in enumerate(test_img):
            with open(IMAGE_DATA_PATH + img, 'r+b') as f:
                with Image.open(f) as image:
                    resized_image = resizeimage.resize_contain(image, target_size)
                    resized_image = resized_image.convert("RGB")
                    #resized_image.save(IMAGE_DATA_PATH + 'resized-' + self.X_train[self.batch_index], image.format)
                    X = img_to_array(resized_image).astype(np.uint8)
                    arr[idx] = X
            if (idx + 1) % 1000 == 0:
                print(idx+1, "out of", len(test_img), "test images have been processed")

        print('\nSaving the processed test images array ... ...')
        print("Size of numpy array = ", sys.getsizeof(arr))
        np.save('../data/processed_test_images.npy', arr)

        print('\nSaving the train/test label arrays ... ...')
        print("Size of training labels numpy array = ", sys.getsizeof(train_lbl))
        np.save('../data/training_labels.npy', train_lbl)
        print("Size of test labels numpy array = ", sys.getsizeof(test_lbl))
        np.save('../data/test_labels.npy', test_lbl)

        pass

    def resize_to_small_images(self, target_size=(299,299), max_qty=None, empty_bins=False):
        '''        
        '''
        # filter out the image files, json files, and labels that exceed the
        # maximum quantity. This is to strip out the the outliers (bin
        # quanities that are too large to detect)
        self._screen_data_by_qty(max_qty, empty_bins)

        # create the processed training image array. Pixel values saved are
        # uint8 to save space. Normalization needs to be done in the model.
        print('\nresize images and save to clean image directory... ...')
        for idx, img in enumerate(self.image_files):
            with open(IMAGE_DATA_PATH + img, 'r+b') as f:
                with Image.open(f) as image:
                    image = image.resize(target_size, Image.ANTIALIAS)
                    image.save(SMALL_IMAGE_DATA_PATH + img, 'JPEG')

            if (idx + 1) % 100 == 0:
                print(idx+1, "out of", len(self.image_files), "images have been processed")

    '''----------------------------------------------------------------------
    Private functions of the ImageProcessing class
    ----------------------------------------------------------------------'''
    def _extract_labels(self):
        '''
        read json files and extract bin qty for each image. These are the
        labels for each image.
        '''
        print("\nExtracting bin quantity labels for each image ... ...")
        start_time = dt.datetime.now()
        labels = []
        for idx, filename in enumerate(self.json_files):
            with open(JSON_DATA_PATH + filename) as f:
                json_data = json.load(f)
                qty = json_data['EXPECTED_QUANTITY']
                labels.append(qty)
        stop_time = dt.datetime.now()
        print("Extracting took ", (stop_time - start_time).total_seconds(), "s.\n")
        return np.array(labels).astype(np.uint8)

    def _get_file_name_arrays(self, num_of_images=1000):
        '''
        Return arrays of image file names and JSON file names, randomly
        shuffled but maintaining consistent order between the two
        '''
        print("\nScanning and shuffling image and json files and ... ...")
        start_time = dt.datetime.now()

        img_list = listdir(IMAGE_DATA_PATH)

        N = 535234
        if len(img_list) < num_of_images:
            N = len(img_list)
        else:
            N = num_of_images

        img_file_list = []
        json_file_list = []
        #print(f"N: {N}")
        for i in range(N):
            if i%1000 == 0:
                print("reading (%d/%d)..." % (i,N))

            jpg_path = '%s%05d.jpg' % (IMAGE_DATA_PATH,i+1)
            jpg_name = '%05d.jpg' % (i+1)
            json_path = '%s%05d.json' % (JSON_DATA_PATH,i+1)
            json_name = '%05d.json' % (i+1)

            #print(f"before: {jpg_path}, {json_path}")

            if os.path.isfile(jpg_path) and os.path.isfile(json_path):
                #print(f"Appended: {jpg_path}, {json_path}")
                img_file_list.append(jpg_name)          
                json_file_list.append(json_name)

        # randomly shuffle the image list and make json list consistent
        new_list = list(zip(img_file_list, json_file_list))
        random.shuffle(new_list)
        img_file_list, json_file_list = zip(*new_list)
        stop_time = dt.datetime.now()
        print("Scanning and shuffling took ", (stop_time - start_time).total_seconds(), "s.\n")
        return np.array(img_file_list), np.array(json_file_list)

    def _get_missing_labels(self):
        '''
        Return the integer quantities missing from the labels
        '''
        start, end = self.unique_labels[0], self.unique_labels[-1]
        return sorted(set(range(start, end + 1)).difference(self.unique_labels))

    def _get_unique_labels(self):
        '''
        Return a sorted list of unique labels
        '''
        return list(sorted(set(self.labels)))

    def _screen_data_by_qty(self, max_qty, empty_bins):
        '''
        The Amazon bin-image data set consists of more than 500k Images
        with quantities that range between 0 and more than 200. This
        functions screens the data to support different counting strategies
        during model development. To start, I will develop a model that
        can recognize empty bins, then I will be trying to count small
        quantities working my way up to larger. This function will screen
        unwanted quanties and ensure there is no class imbalance in the
        remaining images.
        '''
        if empty_bins == True:
            # print(self.image_files)
            # print(self.json_files)
            # print(self.labels)
            mask = np.where(self.labels == 0)
            empty_images = self.image_files[mask]
            empty_json= self.json_files[mask]
            empty_labels = self.labels[mask]
            mask = np.where(self.labels > 0)
            other_images = self.image_files[mask]
            other_json= self.json_files[mask]
            other_labels = np.ones(len(empty_images))
            # now merge. 'other' files are already randomly shuffled so I'll
            # take the first len(empty_images) after masking files with 0 qty.
            all_images = np.append(empty_images, other_images[:len(empty_images)])
            all_json = np.append(empty_json, other_json[:len(empty_images)])
            all_labels = np.append(empty_labels, other_labels)
            # print(all_images)
            # print(all_json)
            # print(all_labels)
            # randomly shuffle the files consistently
            new_list = list(zip(list(all_images), list(all_json), list(all_labels)))
            random.shuffle(new_list)
            all_images, all_json, all_labels = zip(*new_list)
            self.image_files = np.array(all_images)
            self.json_files = np.array(all_json)
            self.labels = np.array(all_labels)
            # print(self.image_files)
            # print(self.json_files)
            # print(self.labels)
        elif max_qty:
            smallest_set = 200000
            groups = []
            for qty in range(max_qty+1):
                mask = np.where(self.labels == qty)
                groups.append([self.image_files[mask],
                               self.json_files[mask],
                               self.labels[mask]])
                if mask[0].size < smallest_set:
                    smallest_set = mask[0].size
            all_images = groups[0][0][:smallest_set]
            all_json = groups[0][1][:smallest_set]
            all_labels = groups[0][2][:smallest_set]
            for idx in range(1, max_qty+1):
                all_images = np.append(all_images, groups[idx][0][:smallest_set])
                all_json = np.append(all_json, groups[idx][1][:smallest_set])
                all_labels = np.append(all_labels, groups[idx][2][:smallest_set])
            # print(all_images)
            # print(all_json)
            # print(all_labels)
            # randomly shuffle the files consistently
            new_list = list(zip(list(all_images), list(all_json), list(all_labels)))
            random.shuffle(new_list)
            all_images, all_json, all_labels = zip(*new_list)
            self.image_files = np.array(all_images)
            self.json_files = np.array(all_json)
            self.labels = np.array(all_labels)
            # print(self.image_files)
            # print(self.json_files)
            # print(self.labels)

            # mask = np.where(self.labels <= max_qty)
            # self.image_files = self.image_files[mask]
            # self.json_files = self.json_files[mask]
            # self.labels = self.labels[mask]

        pass


def main():
    random.seed(39)
    np.random.seed(39)
    tensorflow.compat.v1.set_random_seed(39)
    img_proc = ImageProcessing(3000)
    img_proc.pre_process_images(target_size=(299,299),
                                max_qty=5,    # ignored if empty_bins=True
                                empty_bins=False)


if __name__ == '__main__':
    #main()

    random.seed(39)
    np.random.seed(39)
    tensorflow.compat.v1.set_random_seed(39)
    img_proc = ImageProcessing(19990)
    #img_proc.pre_process_images(target_size=(299,299),
    #                            max_qty=5,    # ignored if empty_bins=True
    #                            empty_bins=False)

    img_proc.resize_to_small_images(target_size=(299,299),
                                max_qty=5,    # ignored if empty_bins=True
                                empty_bins=False)
