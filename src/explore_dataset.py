'''
This script is written to facilitate exploration of the images and associated
json files. The purpose is to gain insights into the data to guide decisions
on model development. Some functions may have value later in the project but
that is not the main goal.
'''
from collections import Counter
from collections import defaultdict
import datetime as dt
import json
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from PIL import Image
import sys

IMAGE_DATA_PATH = '../../data/bin-images/'
JSON_DATA_PATH = '../../data/metadata/'

def get_image_file_names():
    '''
    Return a list of image file names.
    '''
    file_list = [f for f in listdir(IMAGE_DATA_PATH) if isfile(join(IMAGE_DATA_PATH, f))]
    file_list.sort()
    return file_list

def get_json_file_names():
    '''
    Return a list of json file names.
    '''
    file_list = [f for f in listdir(JSON_DATA_PATH) if isfile(join(JSON_DATA_PATH, f))]
    file_list.sort()
    return file_list

def scan_image_files(image_file_list):
    '''
    Return dictionaries containing information and statistics from the
    collection of bin image files in the data set.

    size_dict: qty of items in bin (key) and number of bins with that qty
    '''
    size_dict = defaultdict(int)
    max_width = max_height = 0
    min_width = min_height = 10000
    start_time = dt.datetime.now()
    for i in range(len(image_file_list)):
        with open(IMAGE_DATA_PATH + image_file_list[i], 'r+b') as f:
            with Image.open(f) as image:
                size_dict[image.size] += 1
                width, height = image.size
                if width > max_width:
                    max_width = width
                if height > max_height:
                    max_height = height
                if width < min_width:
                    min_width = width
                if height < min_height:
                    min_height = height
    stop_time = dt.datetime.now()
    print("Elapsed time = ", (stop_time - start_time).total_seconds(), "s.")
    return size_dict, (max_width, max_height), (min_width, min_height)

def scan_json_files(json_file_list):
    '''
    Return dictionaries containing information and statistics from the
    collection of JSON files in the data set.

    item_dict:  class description (key) and the count of the number of times
    that class appears in the list of json files.

    bin_cnt_dict: qty of items in bin (key) and number of bins with that qty
    '''
    item_dict = defaultdict(int)
    bin_cnt_dict = defaultdict(int)
    start_time = dt.datetime.now()
    for i in range(len(json_file_list)):
        with open(JSON_DATA_PATH + json_file_list[i]) as f:
            json_data = json.load(f)
            bin_cnt_dict[json_data['EXPECTED_QUANTITY']] += 1
            for item in json_data['BIN_FCSKU_DATA']:
                item_dict[json_data['BIN_FCSKU_DATA'][item]['asin']] += 1
    stop_time = dt.datetime.now()
    print("Elapsed time = ", (stop_time - start_time).total_seconds(), "s.")
    return item_dict, bin_cnt_dict

def main():
    # test for first command line argument
    if len(sys.argv) < 2:
        print("Provide arguments as follows: \n")
        print("python explore-data.py --sj [-p]")
        print("--sj scan JSON files   [-p] plot results\n")
        print("python explore-data.py --si [-p]")
        print("--si scan JSON files   [-p] plot results\n")
        return

    # lets first get the number of files and file names in each folder
    image_file_list = get_image_file_names()
    json_file_list = get_json_file_names()
    print("First 10 images: \n", image_file_list[:10])
    print("First 10 json docs: \n", json_file_list[:10])

    if sys.argv[1] == '--sj':
        # scan JSON files and extract useful information
        item_dict, bin_cnt_dict = scan_json_files(json_file_list)

        if len(sys.argv) < 3:
            # do nothing - just testing if next arg is null without raising exception
            print("Provide optional -p argument to generate plots of the information.\n")
        elif sys.argv[2] == '-p':
            # create plots of the information extracted from JSON files
            inventory_counts = Counter(item_dict.values())
            fig, ax = plt.subplots(1,1, figsize=(8,4))
            ax.bar(list(inventory_counts.keys()), inventory_counts.values(), width=0.4, color='g')
            ax.set_xlim(0, 20)
            ax.set_xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
            ax.set_ylabel('Number of Unique Items', fontsize=16)
            ax.set_xlabel('Ocurances of Item in Full Dataset', fontsize=16)
            plt.savefig('item_cnts.png', transparent=True)
            plt.show()

            fig, ax = plt.subplots(1,1, figsize=(8,4))
            ax.bar(list(bin_cnt_dict.keys()), bin_cnt_dict.values(), width=0.4, color='g')
            ax.set_xlim(0, 20)
            ax.set_xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
            ax.set_ylabel('Bins', fontsize=16)
            ax.set_xlabel('Items per Bin', fontsize=16)
            plt.savefig('bin_cnts.png', transparent=True)
            plt.show()
    elif sys.argv[1] == '--si':
        # scan image files and extract useful information
        image_size_dict, max_dims, min_dims = scan_image_files(image_file_list)
        print("Maximum image dimensions: ", max_dims)
        print("Minimum image dimensions: ", min_dims)

        if len(sys.argv) < 3:
            # do nothing - just testing if next arg is null without raising exception
            print("Provide optional -p argument to generate plots of the information.\n")
        elif sys.argv[2] == '-p':
            # create plots of the information extracted from image files
            fig, ax = plt.subplots(1,1, figsize=(8,4))
            ax.bar(range(1, len(image_size_dict.keys())+1), image_size_dict.values(), width=0.4, color='g')
            ax.set_xlim(0, 20)
            ax.set_xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
            ax.set_ylabel('Number of Images')
            ax.set_xlabel('Width, Height')
            ax.set_title('Number of Images vs Size')
            plt.savefig('size_cnts.png')
            plt.show()
    pass


if __name__ == '__main__':
    main()
