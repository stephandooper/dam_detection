# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 16:56:01 2019

@author: Stephan
"""
import os
import errno
from google_drive_downloader import GoogleDriveDownloader as gdd


# links updated on 17/12/2019
files = {'water_edges': '1SeQp9l7H9kyqPY3sc_i0FneV3ppRM4fA',
         'random_10':   '1Cq81oV0yenJQAMuv4vQQp-vM8eGHJVAe',
         'random_11':   '1nFMY0lnwNDe3FxiRdDDwoFEurbmtKDDu',
         'random_12':   '1sSn1kyLlpOsv2leuHh--2qG6qdYQh_zv',
         'random_13':   '1ktclyAos6iSMzbkUDQmgcRqCiej5zG20',
         'random_14':   '1VVvtoADMqCeIoHp4auG4cQ1vaUaHmiQo',
         'random_15':   '1ayd7vH0WX8lDQi0jE9-NyS6IuPIyrvJ9',
         'random_16':   '1GVpnX8bRDJmfw8V8oR16vQvLhLTfBCIH',
         'random_17':   '1Z5tWPQzCMMwLheit2s5NX6HzoAnmrxvH',
         'random_18':   '1ra1RD6Lwe7XAHNQrQmcvXK3hQSgToZVv',
         'random_19':   '1u8EMsobIQ3399e9m1_OQuaQnpkF8yA2I',
         'random_20':   '1ipBf89GvwwF3fzYTjujjA0O8-O37CegG',
         'grand'    :   '1IkAoBncRZlyx3SkpK3SFaKsd0KgZDftW',
         'good'     :   '1gnjgnJ2strkOCgKA_fhGwPV0cHVZjItF', 
         'forest'   :   '1K4Dnqvk202pcPp-3bgqJBe-BJ1zFE43l',
         'drip'     :   '1elr_4TGuAfOSh-QvVCLE2bJAv-2_aQxp',
         'bridges'  :   '1oyiAOCf-xaprHYZeWYHU-lyvmtu71GW2',
         'bbox'     :   '1mzHNGxniw-NDto8zlBvArctbc7OK8S7K'
         }
         

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
