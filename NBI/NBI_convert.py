# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 14:44:04 2019

@author: Stephan
"""

import shapefile
from zipfile import ZipFile
import io

# record format can be found here:
# https://www.fhwa.dot.gov/bridge/nbi/format.
# data downloaded from: https://www.fhwa.dot.gov/bridge/nbi/ascii.cfm
# 2018 Data, updated on 22 april 2019
# date accessed: 04/09/2019

FILE_PATH = '../data/bridge_data/2018nodel.zip'

def dmg2Dec(line):
    latDMS = line[129:137] 
    lonDMS = line[137:146]

    # Only process rows with valid lat/lon values
    try:
        if len(latDMS.strip()) > 0 and int(lonDMS.strip()) > 0:
            latDeg = latDMS[0:2] 
            #print latDeg
            latMin = latDMS[2:4]  
            #print latMin
            latSec = latDMS[4:6]+'.'+latDMS[6:8]  
            #print latSec
            latDD = str(int(latDeg)+(float(latMin)/60)+(float(latSec)/3600)) 
            lonDeg = lonDMS[0:3]
            #print lonDeg
            lonMin = lonDMS[3:5] 
            #print lonMin
            lonSec = lonDMS[5:7]+'.'+lonDMS[7:9]  
            #print lonSec
            lonDD = str('-'+str(int(lonDeg)+(float(lonMin)/60)+float(lonSec)/3600)) 
    
            # Do string formatting to lat/long to only display 5 decimal points
            return ('%.5f' % float(latDD)), ('%.5f' % float(lonDD))
    except ValueError:
        # happens when int(lonDMS.strip()) is empty
        print("found empty string inside lonDMS.strip(), cannot cast to int")


def parseLine(line):
    ''' main execution pipeline for a line'''

    try:
        result = []
        
        # get the lat/lon data from dmg
        coords = dmg2Dec(line)
              
        # The year the dam was built       
        # indicates channel/waterway protection     
        # lanes under structure
        # navigational control: filter out bridges not crossing rivers/waterways
        
        result = {'STATE_CODE': line[0:3],
                  'YEAR_BUILT': int(line[156:160]),
                  'LAT': float(coords[0]),
                  'LON': float(coords[1]),
                  #'CHANNEL_PROTECTION': line[261],
                  #'LANES_UNDER_STRUCT': line[162:164],
                  'NAVIGATIONAL_CONTROL': line[188]}
        return result
    
    except (TypeError, AttributeError):
        # dmg2Dec returns None sometimes, in that case, return None...
        print("found nonetype, returning None")
    except ValueError:
        # happens sometimes in the year_built line, when the string is empty
        print("Found empty string inside year_built, can't cast to int")




if __name__ == '__main__':
    """ Parses NBI bridge file and return what we want in a list """
    with ZipFile(FILE_PATH, 'r') as zip:
        # get all filenames inside the zipped directory
        file_list = zip.namelist()
        
        states = []
        for file in file_list:
            # read file (a single state inside the zip) into memory
            # gather all of the data in a single nested list
            items_file = zip.open(file)
            # textiowrapper automatically splits at newlines
            items_file = io.TextIOWrapper(items_file)
            states.append(items_file)
            
            
    # flatten the resulting array
    states = [state for sublist in states for state in sublist]
    
    # map all of the records for every individual state
    states = map(parseLine, states)
    # filter out the results we do not need, i.e. no data
    states = filter(lambda x: x is not None, states)
    # filter out any bridge without any nav control: bridges not over a waterway
    states = filter(lambda x: x['LON'] < -56 and x['LON'] > -173, states )
    states = list(filter(lambda x: x['NAVIGATIONAL_CONTROL'] == '1', states))

    # these commented out give a similar result, but sometimes far away from actual water...
    # since some of these bridges have a very long approach
    #z = list(filter(lambda x: x['CHANNEL_PROTECTION'] != 'N', z))
    #z = list(filter(lambda x: x['LANES_UNDER_STRUCT'] == '00', z))

    # write the records to shapefiles
    w = shapefile.Writer('shapetest', shapeType=1)
    w.autoBalance = 1
    w.field("YEAR_BUILT", 'N' )
    w.field("STATE_CODE", "C")
    for record in states:
        
        w.record(record['YEAR_BUILT'],
                 record['STATE_CODE'])
        
        w.point(record['LON'], record['LAT'])
        
    print(w.recNum == w.shpNum)
    
    w.close()
    print("written shapefile")