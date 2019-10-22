import sys
#sys.path.append("../models/")
sys.path.append("..")
from keras_yolov2.backend import BaseFeatureExtractor
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, BatchNormalization
from tensorflow.keras.applications.resnet50 import ResNet50
from models.darknet19 import darknet19_detection


base_path = './backend_weights/custom/'  # FIXME :: use environment variables

RESNET50_BACKEND_PATH = base_path + "resnet50_backend.h5"  # should be hosted on a server
DARKNET19_BACKEND_PATH = base_path + "darknet19_backend.h5"  

class ResNet50CustomFeature(BaseFeatureExtractor):
    
    def __init__(self, input_size):

        inputs = Input(shape=(input_size[0], input_size[1], 5), name='inputlayer_0')
    
        x = Conv2D(3, (1,1))(inputs)
        resnet50 = ResNet50(weights= 'imagenet', include_top=False)(x)

        self.feature_extractor = Model(inputs=inputs, outputs=resnet50)

        try:
            print("loading backend weights")
            self.feature_extractor.load_weights(RESNET50_BACKEND_PATH)
        except:
            print("Unable to load backend weights. Using a fresh model")
        self.feature_extractor.summary()

class FullYoloCustomFeature(BaseFeatureExtractor):
    
    def __init__(self, input_size):

        params = {'num_classes': 2, 'channels': ['B4','B3','B2','MNDWI','AVE'], 'weights':'imagenet', 'top':False}
        print(params)
        self.feature_extractor = darknet19_detection(**params)

        try:
            print("loading backend weights")
            self.feature_extractor.load_weights(DARKNET19_BACKEND_PATH)
        except:
            print("Unable to load backend weights. Using a fresh model")
        self.feature_extractor.summary()

