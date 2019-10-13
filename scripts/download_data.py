# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 16:56:01 2019

@author: Stephan
"""
import os
import errno

from google_drive_downloader import GoogleDriveDownloader as gdd

files = {'grand': '103pWVUcMhaP0a0ZtwU8CcwKdphf7gYwa', 
         'grand_test':'1y5jkKOgYNwO4t4FWBLdNxdFgynaxtEgD',
         'good':'1jYDmzRS3DbAWvyrrOUPOk6abt55YiRGg',
         'bridges': '1NIkriHjWeeiyEPcKt6o-9oTe8w-lq70A',
         'other': '1XOFrrgGMpBSiRm8vQgiKy4NhWFwhWQ0N'}

def download_file(file_id, dest_path, *args, **kwargs):
    if os.path.exists(dest_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), dest_path)
        
    gdd.download_file_from_google_drive(file_id= file_id,
                                    dest_path= dest_path,
                                    unzip=False,
                                    showsize=True,
                                    overwrite=False)


for file in files.items():
    dest_path = '../datasets/raw/' + str(file[0]) +'.gz'    
    try:
        download_file(file[1], dest_path)
        print("downloading {}  into {} with link {}".format(file[0], dest_path, file[1]))
    except FileNotFoundError:
        print("file {} already exists, not overwriting".format(dest_path))

print("done downloading files")
