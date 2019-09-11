# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 16:56:01 2019

@author: Stephan
"""
import os
import errno

from google_drive_downloader import GoogleDriveDownloader as gdd

1ksH6WZkWOSoh4rZhRT1DAsg7LfWp61ZL
files = {'dams': '1TEgPtclbIaxpSIuva22MvdzTlD6y9BJm', 
         'bridges': '1ksH6WZkWOSoh4rZhRT1DAsg7LfWp61ZL',
         'other': '10lFX8Gr6T6eG_kZ6CVcGBZ_nkE1B5sQd'}

def download_file(file_id, dest_path, *args, **kwargs):
    if os.path.exists(dest_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), dest_path)
        
    gdd.download_file_from_google_drive(file_id= file_id,
                                    dest_path= dest_path,
                                    unzip=False,
                                    showsize=True,
                                    overwrite=False)


for file in files.items():
    dest_path = '../data/raw/' + str(file[0]) +'.gz'    
    try:
        download_file(file[1], dest_path)
        print("downloading {}  into {} with link {}".format(file[0], dest_path, file[1]))
    except FileNotFoundError:
        print("file {} already exists, not overwriting".format(dest_path))

print("done downloading files")
