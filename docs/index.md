# Dam Detection using Neural Networks

Welcome to my blog on dam detection using neural networks. This projects primarily aims to detect dams using satellite data!

## Table of Contents
[Having fun with Earth Engine](#Working with Earth Engine)


*1 August 2019*
## Working with Earth Engine
 
Gathering data to work with has been sort of a challenge. Since I had no prior experience with javascript and Google's documentation on its Earth Engine API is rather minimal, I was mostly left with the tutorials and whatever I could find on the [gis stackexchange](https://gis.stackexchange.com). 

So the basic idea was that I found several datasets with manually annotated dam locations in shapefile format (WGS84) from the [Global Dam Watch](http://globaldamwatch.org/). The dataset I am using for now is called GRanD, which contains over 7000 locations of very large dams. The next logical step is to import this shapefile into Earth Engine. This was rather easy, as I was able to import the necessary shapefiles as an ```asset```
