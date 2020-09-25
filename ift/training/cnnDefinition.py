""" Functional definitions for CNNs explored in LCHb note """

import numpy as np
import tensorflow.keras as keras

__author__ = "Robin Simmons <rs17751@bristol.ac.uk"
def deep_tag_cnn(trackShape, args):

    # two conv layers in series 
    inputFeatures = keras.layers.Input(trackShape)
    
    timeDist = keras.layers.Dense(args.nHidden, activation = 'relu', input_shape = trackShape)(inputFeatures)
    
    drop1 = keras.layers.Dropout(0)(timeDist)
    #8,20
    conv1 = keras.layers.Conv1D(filters = args.filter1, kernel_size = args.kernel1, activation = "relu")(drop1)
    
    conv2 = keras.layers.Conv1D(filters = args.filter2, kernel_size = args.kernel2, activation = "relu")(conv1)
    
    flatten = keras.layers.GlobalAveragePooling1D()(conv2)
    
    dense = keras.layers.Dense(args.nHidden, activation = "relu")(flatten)
    
    drop2 = keras.layers.Dropout(0.8)(dense)
    
    out = keras.layers.Dense(1, "sigmoid")(drop2)
                                  
    return keras.models.Model(inputs = inputFeatures, outputs = out)

def wide_tag_cnn(trackShape, args):

    # two independent conv layers concatenated together 
    inputFeatures = keras.layers.Input(trackShape)
    
    timeDist = keras.layers.Dense(args.nHidden, activation = 'relu', input_shape = trackShape)(inputFeatures)
    
    drop1 = keras.layers.Dropout(0.8)(timeDist)
    
    conv1 = keras.layers.Conv1D(filters = args.filter1, kernel_size = args.kernel1, activation = "relu")(drop1)
    
    conv2 = keras.layers.Conv1D(filters = args.filter2, kernel_size = args.kernel2, activation = "relu")(drop1)
    
    flatten1 = keras.layers.GlobalAveragePooling1D()(conv1)

    flatten2 = keras.layers.GlobalAveragePooling1D()(conv2)

    conc = keras.layers.Concatenate()([flatten1, flatten2])
    
    dense = keras.layers.Dense(args.nHidden, activation = "relu")(conc)
    
    drop2 = keras.layers.Dropout(0.8)(dense)
    
    out = keras.layers.Dense(1, "sigmoid")(drop2)
                                  
    return keras.models.Model(inputs = inputFeatures, outputs = out)

def semi_wide_tag_cnn(trackShape, args):

    # one conv layer concatenated with a dense layer
    inputFeatures = keras.layers.Input(trackShape)
    
    timeDist = keras.layers.Dense(args.nHidden, activation = 'relu', input_shape = trackShape)(inputFeatures)
    
    conv1 = keras.layers.Conv1D(filters = args.filter1 , kernel_size = args.kernel1, strides = args.stride, activation = "relu", padding = args.padding)(timeDist)

    #conv1 = keras.layers.BatchNormalization()(conv1)

    drop1 = keras.layers.SpatialDropout1D(0.4)(conv1)
    
    flatten1 = keras.layers.GlobalAveragePooling1D()(conv1)

    flatten2 = keras.layers.GlobalAveragePooling1D()(timeDist)
    
    #flatten2 = keras.layers.BatchNormalization()(flatten2)
    
    conc = keras.layers.Concatenate()([flatten1, flatten2])

    dense = keras.layers.Dense(args.nHidden, activation = "relu")(conc)   

    drop2 = keras.layers.Dropout(0.6)(dense)
    
    out = keras.layers.Dense(1, "sigmoid")(drop2)
                                  
    return keras.models.Model(inputs = inputFeatures, outputs = out)

def attention_tag_cnn(trackShape, args):

    # two conv layers combined with attention and concatenated with a dense output
    inputFeatures = keras.layers.Input(trackShape)
    
    timeDist = keras.layers.Dense(args.nHidden, activation = 'relu', input_shape = trackShape)(inputFeatures)
    
    drop1 = keras.layers.Dropout(0.2)(timeDist)
    
    conv1 = keras.layers.Conv1D(filters = args.filter1, kernel_size = args.kernel1, activation = "relu")(drop1)
    
    conv2 = keras.layers.Conv1D(filters = args.filter2, kernel_size = args.filters2, activation = "relu")(drop1)

    atten = keras.layers.Attention()([conv1, conv2])
    
    flatten1 = keras.layers.Flatten()(atten)

    flatten2 = keras.layers.Flatten()(drop1)

    conc = keras.layers.Concatenate()([flatten1, flatten2])
    
    dense = keras.layers.Dense(args.nHidden, activation = "relu")(conc)
    
    drop2 = keras.layers.Dropout(0.3)(dense)
    
    out = keras.layers.Dense(1, "sigmoid")(drop2)
                                  
    return keras.models.Model(inputs = inputFeatures, outputs = out)

def cnn_simple(inputShape, args):
     
    inputFeatures = keras.layers.Input(trackShape)
    
    timeDist = keras.layers.Dense(args.nHidden, activation = 'relu', input_shape = trackShape)(inputFeatures)
    
    drop1 = keras.layers.SpatialDropout(0.1)(timeDist)
    #8,20
    conv1 = keras.layers.Conv1D(filters = args.filter1, kernel_size = args.kernel1,
                                          activation  = "relu")(drop1)
        
    flatten = keras.layers.GlobalAveragePooling1D()(conv1)
    
    dense = keras.layers.Dense(args.nHidden, activation = "relu")(flatten)
    
    drop2 = keras.layers.Dropout(0.8)(dense)
    
    out = keras.layers.Dense(1, "sigmoid")(drop2)
                                  
    return keras.models.Model(inputs = inputFeatures, outputs = out)

