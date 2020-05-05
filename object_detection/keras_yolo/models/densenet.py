#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 13:19:53 2019

@author: stephan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 14:05:17 2019

@author: stephan
"""
from tensorflow.keras.layers import Conv2D, Flatten, Input, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications.densenet import DenseNet121

def build_densenet121(train_model=True, **kwargs):
    inputs = Input(shape=(None, None, len(kwargs.get('channels')) ), name='inputlayer_0')
    weights = kwargs.get('weights')

    if weights is None:
        print("instantiated with random weights")
        densenet121 = DenseNet121(input_tensor=inputs, weights=weights, include_top=False)
        densenet121 = densenet121.output
    else:
        print("instantiated model with weights: {}".format(weights))
        #dense_filter = Conv2D(filters=3, kernel_size=(3,3), padding='same')(inputs)

        x = Conv2D(3, (1,1))(inputs)
        densenet121 = DenseNet121(weights=weights, include_top=False)(x)
    
    if kwargs.get('top') == False:
        # top is not needed for object detectors
        model = Model(inputs=inputs, outputs=densenet121)
    else:
        output = GlobalAveragePooling2D()(densenet121)
        output = Dense(kwargs.get('num_classes'), activation='softmax')(output)
        model = Model(inputs=inputs, outputs=output)

    model.summary()
    return model
