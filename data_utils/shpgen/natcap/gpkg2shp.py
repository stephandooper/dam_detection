# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 15:22:03 2019

@author: Stephan
"""

import geopandas

df = geopandas.read_file('natcap_bbox.gpkg')

df.to_file('natcap_bbox.shp', driver='ESRI Shapefile')