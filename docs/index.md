# Dam Detection using Neural Networks

Welcome to my blog on dam detection using neural networks. This projects primarily aims to detect dams using satellite data!

## Table of Contents
[Some segmentation maps](#segmentation-maps)
[Getting reproducibility in Tensorflow GPU](#getting-reproducibility-in-tensorflow-gpu)   
[Logging experiments with Sacred and Omniboard](#logging-experiments-with-sacred-and-omniboard)   
[Shuffle, shuffle, shuffle](#shuffle-shuffle-shuffle)  
[A problem with elevation](#a-problem-with-elevation)   
[Working with TFRecords](#working-with-tfrecords)   
[Exploring Earth Engine](#exploring-earth-engine)   
1. [An update](#an-update-10082019)

*1 November 2019
## Segmentation maps
After many hours of figuring out how to export large images in GEE using the ```export.Image.ToDrive``` (featurecollections have a limit on the size), I was finally able to create the first heatmaps of the models that I trained. In essence, a large image is cut into several patches, and sometimes even be distributed among multiple TFRecords. The information on how to put them all back together is saved in a mixer.json file. After doing that, the resulting heatmaps from a dilated fcn looks like this:

![](images/original.png)
![](images/heatmap.png)

Although I was able to get a fairly high test score on single classification labels, the heatmaps seem to accent on the contours of objects, rather then the dam itself. This means that there definitely is more work being done, which is not altogether bad, considering we can see this as a starting point.  

* 11 October 2019*
## Getting reproducibility in Tensorflow GPU
Lately I have been training several models and I noticed that my losses and metrics were not producing the same result in Tensorflow with GPU and with Tensorflow Keras (not the standalone Keras). A quick ablation analysis lead me to believe the main culprit was the GPU: after disabling the GPU and training on the CPU, I found that my experiments were reproducible, but when GPU was involved it was not. This is not entirely strange though, since GPU computations are incredibly complex, and ensuring determinism is a daunting task. I found that there were several culprits, such as the ```tf.math.reduce_sum``` and ```tf.math.reduce_mean``` not being deterministic, despite having set a seed. 

Naturally, I started looking for a solution, but I could not find anything useful, other then [disabling the GPU entirely](https://stackoverflow.com/questions/45230448/how-to-get-reproducible-result-when-running-keras-with-tensorflow-backend), which I do not really constitute as a fix. However, there now is a fix in the [tensorflow-determinism](https://github.com/NVIDIA/tensorflow-determinism) package. All I had to do is install it and add a single line of code (given in the docs) and I was done. Reproducibility at last!

* 30 September 2019
## Getting more data
It has been a while since I posted anything dam-related. Most of my posts so far have only been about pipeline creation and some design decisions. Although it is still too early to give some analysis results (or at least I feel they are not really all that interesting yet), I think I can write some more about the data sources and its distribution. 

When I started this project, I had two data sources: the first one was the GRanD dataset which was a manually georeferenced dataset of dam locations with about 7300 records, and a smaller dataset with 1000 records of non-dam locations near water edges that were generated in GEE, see [Charlotte Weil's blog post](https://medium.com/@charlotteweil/can-we-locate-dams-from-space-2a796ac8c04b). I knew that this was not nearly enough data to constitute a representative dataset of (non) dam locations, so I definitely needed more. The major cases here were:

1. I could still get more data from the [Global Dam Watch](http://globaldamwatch.org/), especially the GOOD<sup>2</sup> dataset is interesting since it contains over 30000 dams, both large and medium sized. There are also some secondary sources, such as the U.S. NDI dataset, which features dams in the United Stated.
2. Obtaining bridge data could also come in handy. At this time I am not sure whether the neural network would be able to keep apart bridges from dams, so this seems like a nice hypothesis to test. But in order to do this, I would need data on bridges. I was quite disappointed to find that there are not all that many datasets available, except for the extensive [NBI dataset](https://www.fhwa.dot.gov/bridge/nbi.cfm). The data are simply a set of (non)-delimited ASCII files, and have to be converted in shapefiles manually, which I did [here](https://github.com/stephandooper/dam_detection/tree/master/NBI).
3. Getting other non-dam locations. I was not satisfied with only having 1000 negative samples to traing with. Since I had access to GEE, I could practically obtain an unlimited amount of non dam samples, which I will explain further below.

### Sampling near water edges
Charlotte's blog post did say that they sampled the negative examples in a smart way by sampling in the neighborhood of water edges (locations where water meets land), since they could be hard to learn. The only problem is that they did not write *how* they did it other than they used the [JRC GSW (v1.1)](https://developers.google.com/earth-engine/datasets/catalog/JRC_GSW1_1_GlobalSurfaceWater) layer, so this took some time to figure out, and I created my own solution in GEE to do this, which are outlined in the following steps:

1. I started by selecting the *occurrence* band from the JRC GWS layer, which displays water occurrence across continents.
2. Then, I proceeded by using a [Canny edge detector](https://en.wikipedia.org/wiki/Canny_edge_detector) on the occurrence band, and masked the result with itself. 

The code to reproduce this is given below, along with an extra steps to remove large values from the edge detector, since there were some very thich edges along the coasts of the continent that I did not want to sample from.

```javascript
// Detect edges in the occurence image
//Map.addLayer(occurrence,VIS_OCCURRENCE,'updated water');
var canny = ee.Algorithms.CannyEdgeDetector(occurrence, 60, 1);

// Mask the image with itself to get rid of areas with no edges.
// border have value occurence, non border values are 0
canny = canny.updateMask(canny);


// Remove outer boundary in the ocean (outer boundary near continents)
var newmask = canny.gte(220).not();

Map.addLayer(canny.updateMask(newmask), {min: 0, max: 150, palette: 'FF0000'}, 'canny edges');
Map.addLayer(canny, {min: 0, max: 150, palette: 'FF0000'}, 'canny edges no mask');

canny = canny.updateMask(newmask);
```

Then, I simply defined several regions of interest, and sampled along the Canny edges inside of my ROI (example only includes the USA, but in reality I did define more regions):


```javascript
function sampleRegion(region, factor){
  var sampledRegion = canny.sample(
    { region: region, 
      geometries: true, 
      scale: 100, 
      factor: factor, 
      tileScale: 1,
      seed:1
    });
  return sampledRegion;
}

// --> high factor, high scale
// US
var sampledUS = sampleRegion(USA, 0.0002);
// compute the size of the sampled points
print("# points US", sampledUS.size());
Map.addLayer(sampledUS,{color: "blue"},'US samples');
```

### Other non dam locations
Since water edges are not the only negative samples that should be considered, I also sampled random points over all continents by using the following code

```
var sampledRoi = ee.FeatureCollection.randomPoints(roi,numSample, 0, SEED);
Map.addLayer(sampledRoi, {color: 'orange'}, 'roi');

```


*16 September 2019*
## Logging experiments with Sacred and Omniboard
During my medical imaging course, I was introduced with [Sacred](https://github.com/IDSIA/sacred) and [Omniboard](https://github.com/vivekratnavel/omniboard), and I felt it would be a good idea to use them for this project as well. The former is a tool that can be used to log, organize, and even help reproduce experiments, whereas the latter is a web dashboard to visualize the data logged in Sacred. The way it works is that we can set up a [MongoDB](https://www.mongodb.com/) instance, which serves as the databased where all of the configuration data and logs will be stored from Sacred. The exact way on how to connect Sacred with a MongoDB instance is explained in the [Sacred docs page](https://sacred.readthedocs.io/en/stable/observers.html#mongo-observer). There are also several features for authentification. 

Once we have a Sacred instance running with MongoDB, we still cannot do much without any decent organized visualization of all the data that we stored, so this is where Omniboard comes in. Setting up an Omniboard instance on a local machine is not really that hard using the [quick start](https://vivekratnavel.github.io/omniboard/#/quick-start) instructions. In fact, after using ```pip``` to install the required packages, all that needs to be done is to run the omniboard instance with the following codes, replacing the strings in "<>" with your own information:

```cmd
omniboard --mu "mongodb://<username>:<password>@<host>/<database>[?options]"
```
However, it is quite tiresome to keep loading omniboard on a local machine as part of a start up routine. Instead, it would be easier to set up once on a dedicated server, at which point it would run constantly. This makes more sense as Omniboard is a *web* dashboard. Finding such a dedicated server for *free* was a bit of a hassle, since most web services like Microsoft Azure, Amazon Web Services (AWS), and Google Cloud Platform (GCP) require you to pay for their services, give only a free trial on the instances that are needed, or ask some kind of payment information which I did not want to provide. Fortunately I came across another well known hosting service: [Heroku](https://www.heroku.com/), which allowed me to host (node) JS apps for free. 

Setting up the Omniboard instance on Heroku also proved to be difficult: when I tried building on the platform itself, I was riddled with dependency and package errors. Since I was not familiar with the ```npm``` or javascript related problems at all (besides GEE) I hit a wall, and admitted defeat (thus continued to run Omniboard locally). 

Luckily, I saw later that Heroku supports [Docker](https://www.docker.com/) instances, and I recalled using it during my Big Data course. At the time I was told that it is basically a way to run a program inside a container without any hassle. Back then, I did not really see the point in using it, and it seemed to just take up more time instead, as you would have to take into account the overhead of starting up the entire docker instance. Now, however, I saw that it could be very useful: all I had to do was build Omniboard myself just once succesfully, and push the resulting Docker image to Heroku, and to make things even better, Omniboard already has an [available Dockerfile](https://hub.docker.com/r/vivekratnavel/omniboard/dockerfile). The only things that were left were: (re)install docker, npm to build Omniboard, git clone Omniboard onto my desktop, copy the Dockerfile, and put it all together. I did have to add an extra CMD argument to the Dockerfile at the end to run it with the --mu parameter and authorization, which looked like this (ofcourse, the username, password, and database name were changed in this example):

```
CMD ["--mu", "mongodb+srv://my-username:my-password@my-cluster-v9zjk.mongodb.net/sacred?retryWrites=true"]
```

All that is left is to build a working Omniboard instance inside Docker, and simply push it to Heroku (a tutorial on how to do this can be found in the [Heroku docs](https://devcenter.heroku.com/articles/container-registry-and-runtime). Once I built the image and pushed it to container registry, I was done, and I finally had a running container on Heroku, which I can access at any time without having to start Omniboard myself every single time I want to work on my project. The dashboard looks something like this:
![](images/omniboard1.png)

### About Sacred Reproducibility 
An important thing to know is that it is possible to set a general seed using Sacred, which simplifies reproducibility since it will basically 'take care of everything' for you. The only downside is that this does not hold for tensorflow. Although Sacred will be able to set a seed for ```random``` and ```numpy```, it seems tensorflow reproducibility still needs to be done the old fashioned way.



*2 September 2019*
## Shuffle, Shuffle, Shuffle
Last time I covered a quick way to define a data transformation pipeline. At that moment I stil only had a single, large TFRecord that I had to parse. After reading some [discussions](https://datascience.stackexchange.com/questions/16318/what-is-the-benefit-of-splitting-tfrecord-file-into-shards/22767), I came to the conlusion that having a single 4GB TFRecord might not be the best option. For now I will explain why I think this is not best way to go, and how I solved it. 

First, it might be a good idea to explain how shuffling in TFRecords actually works [(Tensorflow docs)](https://www.tensorflow.org/api_docs/python/tf/data/Dataset): if your dataset contains 10,000 elements but the buffer_size is set to 1,000, then shuffle will initially select a random element from only the first 1,000 elements in the buffer. Once an element is selected, its space in the buffer is replaced by the next (i.e. 1,001-st) element, maintaining the 1,000 element buffer.

This design was implemented so that data could be shuffled in memory, even with potentially huge datasets. In the end, the amount of memory necessary is tuned via the buffer size parameter. A consequence of the shuffle-queue design is that the shuffle function only allows for local randomness (inside of the queue, not across the entire record). As such, if the buffer size is not sufficiently large, then the sampled distribution will approximately have a relation with the original order. An illustration is shown below. The shuffle buffer is filled with elements in deterministic order. Then, an element is chosen at random and is processed further.

![](images/shuffle_diagram.png)

This is important to know, because backpropagation benefits from diversified samples. Ideally, you would feed a batch with a diverse set of classes. Furthermore, when you are passing the data a next time (a new epoch), you would want the data in a completely different order from the previous epoch, since this also helps with backpropagation.
In order to get a better sampling procedure, it might thus be a good idea to set a very high buffer size, but this has its limits because the buffer has to be filled, and is limited to the amount of available memory. As a compromise, it might be better to split up my one large TFRecord into multiple smaller ones. This is done in [TFRFilter.py](https://github.com/stephandooper/dam_detection/blob/master/scripts/TFRFilter.py), which also removes any corrupted records or empty data slots. The final implementation then consists of using a block shuffle strategy, along with the regular shuffle method, depicted below:

![](images/block_shuffle.png)


This new way of shuffling gives me far more freedom to shuffle the data that I have. 

*22 August 2019*
## A problem with elevation
While debugging the pipeline I found that I kept getting several errors while parsing through the TFRecords. Most notably I was stuck on this specific one:

```python
Feature: B2 (data type: float) is required but could not be found. [[{{node ParseSingleExample/ParseSingleExample}}]] [Op:IteratorGetNextSync].
```

The error message was actually quite clear for once: there was no B2 (blue) channel for a specific example inside of the TFRecord, but the thing that confused me the most was: why was it not there? Although it took some time, I finally realized it was because there was no satellite data available at the specific location and date range where I was rendering the image patches. The easiest solution was to increase the date range in the ```filterDate``` for the Sentinel 2 images, but this did not really solve the bigger problem with another data band I was using: the elevation band (ALOS DSM 30m). It turns out that this band has a lot of missing data in the northern hemisphere due to clouds, snow, etc. You can see this in the image below

![](images/dsm.png)

It turns out that there are many stripes where elevation data is missing, and there are several dam locations in those areas where data is missing, which explains the error. At this point I had no other choice but to filter out these dam locations if I wanted to use elevation, which is what I ended up doing. Luckily, there were not many dam locations that needed to be filtered and the impact seems minimal, but it might be a hurdle in the future if someone wants to run an anlysis in the northern hemisphere.

*15 August 2019*
## Working with TFRecords
TLDR: [Jupyter Notebook](https://github.com/stephandooper/dam_detection/blob/master/dam_detection.ipynb)   
After my last post, I was able to extract all of the images that I needed (GRanD dam locations, and some non-dam locations from [Charlotte Weil's blog post](https://medium.com/@charlotteweil/can-we-locate-dams-from-space-2a796ac8c04b). I was now finally ready to build a pipeline to parse the data and train some initial networks. The only problem was: the data was locked in TFRecords, and I had never worked with them before. This meant that I first had to read some tutorials on how to work with TFRecords. 

Basically, TFRecords work with a protocol buffer framework. Protocol buffers provide a mechanism to read and store data in an efficient way. What this means for us in practice is that we can work with potentially huge datasets, even if we do not have sufficient memory. After reading some tutorials I came up with a small list of the most important (dis)advantages

***Advantages***
1. The TFRecord data pipeline allows us to access huge amount of datasets, even if we do not have sufficient memory to load the entire dataset into, as explained above. All we need is enough persistent storage, and we are good to go.
2. We can make use of the ```Dataset``` API, along with Tensorflow eager execution. This has made things significantly easier from the old school sessions-based approach. ```Dataset``` is an iterable object, which also has some very straightforward functions, of which the most important is ```Dataset.map()```, which basically maps a user-written function over several workers. Some other useful functions are ```batch, shuffle, interleave, list_files, prefetch, repeat```. It also supports user-defined lambda expressons.
3. Parallelization & optimization: working with a TFRecord pipeline (and the Dataset API) allows us to parallelize and optimize the training procedure quite easily. Basically, the ```map``` function allows us to distribute the pre-processing across multiple workers (there is even a Spark application, which I do not use). Also, the CPU bottleneck has been addressed by creating a data pipeline that makes sure that the idle time of the CPU and GPU is minimzed (they basically improved the scheduling), see [Tensorflow docs](https://www.tensorflow.org/guide/data_performance). All of this allows us to gain a huge boost in training speed, if used correctly. See for example [speeding up keras with tfrecord datasets](https://medium.com/@moritzkrger/speeding-up-keras-with-tfrecord-datasets-5464f9836c36), or [optimize tf input pipeline](https://sebastianwallkoetter.wordpress.com/2018/02/24/optimize-tf-input-pipeline/), which covers performance within a TFRecord pipeline

***Disadvantages***
1. The pipeline has a steep learning curve and takes some to implement. Since it is not as straightforward as loading a folder of images into memory, it takes a while to grasp the idea of why the TFRecord pipeline is useful, and how to use it properly. 
2. Since we are using a Tensorflow pipeline, we sadly do not have access to several packages which make life easier. Basically, we are stuck with whatever functionality Tensorflow offers, and cannot go outside it (without explicitly converting to numpy and suffering a major performance penalty). An example of this is the ```imgaug``` package, which provides a huge number of image augmentations, which you can define in a wide variety in a flexible way. As such, we will have to create our own image augmentation and augmentation routine (when to apply which augmentation, and in which frequency) using only Tensorflow.

#### Creating the pipeline
Surely, I was only able to make up this list after having played around with the TFRecord pipeline and trying to create one myself. Admittedly , this took longer than I expected. The framework is quite difficult to learn at first, especially since you have to write several lines of code before you can get some interpretable outputs, in this case being able to view the images. Since I do not want to bother anyone with a whole tutorial on how to parse TFRecords I will simply just refer to the [Jupyter Notebook](https://github.com/stephandooper/dam_detection/blob/master/dam_detection.ipynb) that I created.

While I was working my way through the entire process of finally being able to view the images that were locked away inside of the TFRecords, I noticed that they appeared to be very dark. The apparant reason for this was that the images color channels were in the range of [0, 0.3], whereas python expects either a range of [0,1] or [0,255]. A quick fix was to take the global maximum over the entire image (all 3 color channels) and divide each element in the channels by this number. The results was that the image itself is now in a range of [0,1], and looks a lot brighter, as shown below. 

![](images/dams.png)


Now that I am able to visualize the images inside the TFRecords, I thought it would be time to creat a small network. the Dataset API expects me to define a training routine beforehand. This means that I will have to list the operations I wish to perform on the TFRecords before passing it to the Tensorflow Keras API. An example of how this is done is shown below. Keep in mind that this is a code snipper from the notebook, and can be different later on, as I do not intend to update the code snippets inside of the blog as they could become too complicated.

```python
def create_dataset(file_names, target_size, buffer_size, channels, stretch_colorspace=True, augmentations=[]):
    files = tf.data.Dataset.list_files(file_names, shuffle=None, seed=SEED)
    
    dataset = tf.data.TFRecordDataset(files, compression_type='GZIP')
    dataset = dataset.shuffle(buffer_size=buffer_size)

    dataset = dataset.map(parse_serialized_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(parse_image(dims = target_size, channels = channels, stretch_colorspace=stretch_colorspace), 
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(32)

    return dataset
```
What this code does is actually quite simple. First, we pass the location(s) of our TFRecords into the ```list_files``` function. At the time of writing, I only had a single (large) TFRecord, obtained from GEE. Then, a Dataset object is created from the ```tf.data.TFRecordDataset``` function. Now we can easily create a pipeline of operations that have to be performed before feeding the data into the model. We start by shuffling the records, since the TFRecords are parsed in a first in first out order, and are therefore deterministic. The ```shuffling``` loads the first ```buffer_size``` elements and randomly selects one of the elements to be processed first, thus randomizing the order. 

Then, we apply two map functions. The first one parses the TFRecords and converts them into a dictionary as key value pairs, which is then used in the second map function, where the images are parsed. Finally, we can create batches using the ```batch``` function, and we are now able to pass it to a keras model, which can be found in the notebook.


[Working with TFRecords](https://towardsdatascience.com/working-with-tfrecords-and-tf-train-example-36d111b3ff4d)

*1 August 2019*
## Exploring Earth Engine
 
Gathering data to work with has been sort of a challenge. Since I had no prior experience with javascript and Google's documentation on its Earth Engine API is rather minimal, I was mostly left with the tutorials and whatever I could find on the [gis stackexchange](https://gis.stackexchange.com). 

So the basic idea is that I found several datasets with manually annotated dam locations in shapefile format (WGS84) from the [Global Dam Watch](http://globaldamwatch.org/). The dataset I am using for now is called GRanD, which contains over 7000 locations of very large dams. The next logical step is to import this shapefile into Earth Engine. This was rather easy, as Google Earth Engine allows you to import external files as ```assets```, and import them as featurecollections in the code editor. Since I am completely new to GIS software the easiest thing to do is to visualize the dam locations, which looks like this:
![](images/grand_dams.png)

Eyeballing the locations the coordinates seem to be in the right places, altough a little far off from the actual dam locations at times, but we'll just have to live with that. The next step is that I want to extract and download image patches around these coordinates so that I can feed them into a classifier for training. This turned out to be more difficult than I wanted, as the Earth Engine docs did not explicitly cover this. Luckily I found a blog post by [Charlotte Weil](https://medium.com/@charlotteweil/can-we-locate-dams-from-space-2a796ac8c04b) covering the same topic. In order to get to the result that we want we have take several steps, explained below.

The first step is to choose a satellite and the layers you want to use. For this project, I used the [Sentinel 2](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2) Multispectral Instrument (MSI) with Level-1C orthorectified top-of-atmosphere reflectance. The most common problem with using satellite images is that they can be obstructed by clouds. Luckily GEE provides a code that removes most of the clouds right off the bat:

```javscript
function maskS2clouds(image) {
  var qa = image.select('QA60');

  // Bits 10 and 11 are clouds and cirrus, respectively.
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;

  // Both flags should be set to zero, indicating clear conditions.
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0).and(
             qa.bitwiseAnd(cirrusBitMask).eq(0));

  return image.updateMask(mask).divide(10000).select('B.*').copyProperties(image, ["system:time_start"]);
}

// Map the function over one year of data and take the median.
var dataset = ee.ImageCollection('COPERNICUS/S2_SR')
                  .filterDate('2018-01-01', '2019-05-15')
                  // Pre-filter to get less cloudy granules.
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                  .map(maskS2clouds);

var reduce_dataset = dataset.median();
var rgbVis = {min: 0.0,  max: 0.3,  bands: ['B2'],};

// display masking, and rgb
Map.addLayer(reduce_dataset.select('B2'), rgbVis, 'RGB');
```

The next step is to extract the bands (channels) we would like to use for training the classifier. The bands we choose are the RGB bands, which are encoded as B4, B3, and B2 in S2, and compute the Normalized Difference Water Index (NDWI), for which we also need the Near Infrared (NIR) band, which is named B8 in S2. All of the bands used from the S2 satellite are sampled at 10m resolution. Following the blog post from Charlotte, elevation is also factored in by using the Alos DSM. This brings up to a total of 5 bands (channels) to use for training. The code for selecting, calculating, and adding the RGB, NDWI, and elevation bands is given below.

```Javascript
// Calculate NDWI
var ndwi = reduce_dataset.normalizedDifference(['B3', 'B8']).rename('NDWI');
// use the imported ALOS DSM map and select the AVE band (average elevation)
var elevation = DSM.select(['AVE']).float();

// add bands, and sample in a neighborhood near the coordinates
var combinedImage = reduce_dataset.addBands(ndwi).addBands(elevation);
```
The next step is a crucial one and not in any of the official GEE tutorials. In order to extract pixels in a neighborhood around the GRanD dam coordinates, we can use the ```neighborhoodToArray``` function, which does exactly what we need: It turns the neighborhood of each pixel in a scalar image into a 2D array (see image below). Axes 0 and 1 of the output array correspond to Y and X axes of the image, respectively. The output image will have as many bands as the input; each output band has the same mask as the corresponding input band. The footprint and metadata of the input image are preserved.

![](images/neighborhoodToArray.png)

The GEE implementation of this functionality looks like this:
```javascript
var sampleImage = combinedImage.neighborhoodToArray(ee.Kernel.rectangle(128,128, 'pixels')).select(['B2','B3','B4','NDWI','AVE']);
```
Note that we first have to define a shape (a rectangular kernel), so that the neighbohoodToArray function knows in which actual neighborhood to extract the pixels. A rectangle format of 128x128 pixels was chosen, which yields an image of 257x257.   

We are almost there, but we still need an extra step before we are able to export the image patches. Right now we have defined a sampleImage variable, but this is still an image feature with 5 bands. However, we do not want to export an image, but rather a more organized table, where each record contains an image patch from the GRanD locations. So what we want is a **featurecollection**, which has as many records as there are GRanD dam locations, and where each record holds the 5 bands as 2D arrays. This can be achieved using the ```sampleRegions``` function, which is outlined below:

```javascript
var dams_neighborhoods = sampleImage.sampleRegions({
            collection:grand,
            properties:['DAM_NAME'],
            scale:10,
            tileScale:4,
        }).map(function(feature) { return feature.set({class: 1}); });
```
This function takes the sampleImage we have defined above, and samples patches of 257x257 as defined in the neighborhoodToArray function, at the locations specified in the grand collection (so in a neighborhood of 128x128 around each grand dam location). Note that we also add a feature to each example in the featurecollection.

We now have a featurecollection with all of the desired data, that we can now export. This is by far the easiest part, and can be done as shown below

```javascript
Export.table.toDrive({
  collection: dams_neighborhoods,
  folder:'Dams',
  description: 'Grand_Dams',
  fileFormat: 'TFRecord',
  fileNamePrefix:'GrandDamSamples',
  selectors: ['label', 'B2', 'B3', 'B4','AVE','NDWI']
});
```

This all took quite a while to figure out, but it did pay off.

### An update 10/08/2019
Although sampleRegions is a convenient function, it does not allow for very large images to be sampled. As such, I was riddled with computation time out errors. Luckily, I was able to find a workaround. The key idea is that sampleRegions uses image.reduceRegions in its computations, which tries to do all of the computations in a single go. Since the amount of images that need to be extracted is large, it will never be done before the computation time limit is exceeded. To circumvent this problem, we can use image.reduceRegion instead, which computes only a single coordinate. We can simply put a map with the featurecollection as data input to compute all of the image patches. This way, the computation timer resets for each individual image. The only problem is that it might be slower than the sampleRegions function, which you can read about on [GEE](https://developers.google.com/earth-engine/debugging) 

```
// Sample grand dams
var damSamplesGrand = grand.map(function(feature) {
  // alternative to sampleRegion
  return feature.set(sampleImage.reduceRegion({
    reducer: 'first',
    scale:10,
    geometry: feature.geometry(),
    tileScale:4
  })).select(['B2', 'B3', 'B4', 'AVE', 'NDWI', 'DAM_NAME']);
}).map(function(feature){return feature.set({label:1})});
```


