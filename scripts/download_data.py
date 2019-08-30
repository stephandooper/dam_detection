# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 16:56:01 2019

@author: Stephan
"""
import os
cwd = os.getcwd()
print(cwd)

from google_drive_downloader import GoogleDriveDownloader as gdd


gdd.download_file_from_google_drive(file_id='1rBgrHwfvoEVjrhs8M41yzIiy6vDIpYAi',
                                    dest_path=os.path.join(cwd,'waterdges.zip'),
                                    unzip=True,
                                    showsize=True)

print("done?")