# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 22:38:35 2019

@author: Stephan
"""

import shapefile as shp
import numpy as np
import pandas as pd
from shapely.geometry import Point
import geopandas

df = pd.read_csv('NID2018_U.csv')
df = df[df['LONGITUDE'].notnull()]
df = df[df['LATITUDE'].notnull()]
df = df[df['LONGITUDE'] < -15]
df = df[df['LATITUDE'] > 0]
f = df['LATITUDE']

df['geometry'] = df.apply(lambda x: Point((float(x.LONGITUDE), float(x.LATITUDE))), axis=1)
df = geopandas.GeoDataFrame(df['geometry'], geometry='geometry')
df.to_file('NID.shp', driver='ESRI Shapefile')
