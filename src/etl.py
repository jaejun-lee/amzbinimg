import boto3
import os
import timeit

BUCKET_NAME = 'aft-vbi-pds'
LIST_FILE = '../data/list_s3_objects.txt'
BIN_IMAGE_DIR = '../data/bin-images/'
METADATA_DIR = '../data/metadata/'

def download_objects(start=0, end=10):
    '''retrieve Amazon Bin Images Dataset

    Parameters:
        start: int
        end: int

    TODO: add script to populate list_s3_objects.txt
    '''
    s3 = boto3.client('s3')

    with open(LIST_FILE, 'r') as fp:
    
        for i in range(0, start):
            fp.readline()
        
        for i in range(start, end):
            str = fp.readline()
            image_obj = str.split()[3]
            metadata_obj = f"{image_obj.split('.')[0]}.json"
            image_filepath = f"{BIN_IMAGE_DIR}{image_obj}" 
            metadata_filepath = f"{METADATA_DIR}{metadata_obj}" 
            image_obj = f"bin-images/{image_obj}"
            metadata_obj = f"metadata/{metadata_obj}"
            print(f"downloading {image_obj}")
            s3.download_file(BUCKET_NAME, image_obj, image_filepath)
            print(f"downloading {metadata_obj}")
            s3.download_file(BUCKET_NAME, metadata_obj, metadata_filepath)

if __name__=="__main__":
    pass
    #downloaed from 0 to 1000 object
    download_objects(10000, 20000)

