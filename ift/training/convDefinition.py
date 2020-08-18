import numpy as np

import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow import dtypes, matmul, Tensor, tensordot, transpose
import tensorflow_model_optimization as tfmot 

LastValueQuantizer = tfmot.quantization.keras.quantizers.LastValueQuantizer
MovingAverageQuantizer = tfmot.quantization.keras.quantizers.MovingAverageQuantizer
quantize_annotate_layer = tfmot.quantization.keras.quantize_annotate_layer
quantize_annotate_model = tfmot.quantization.keras.quantize_annotate_model


class DefaultDenseQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    def __init__(self, bits = 8):
        self.bits = bits
        
    # Configure how to quantize weights.
    def get_weights_and_quantizers(self, layer):
      return [(layer.kernel, LastValueQuantizer(num_bits = self.bits, symmetric=True, narrow_range=False, per_axis=False))]

    # Configure how to quantize activations.
    def get_activations_and_quantizers(self, layer):
      return [(layer.activation, MovingAverageQuantizer(num_bits = self.bits, symmetric=False, narrow_range=False, per_axis=False))]

    def set_quantize_weights(self, layer, quantize_weights):
      # Add this line for each item returned in `get_weights_and_quantizers`
      # , in the same order
      layer.kernel = quantize_weights[0]

    def set_quantize_activations(self, layer, quantize_activations):
      # Add this line for each item returned in `get_activations_and_quantizers`
      # , in the same order.
      layer.activation = quantize_activations[0]

    # Configure how to quantize outputs (may be equivalent to activations).
    def get_output_quantizers(self, layer):
      return []

    def get_config(self):
      return {}

class CustomDenseQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    def __init__(self, bits = 8):
        self.bits = bits
        
    # Configure how to quantize weights.
    def get_weights_and_quantizers(self, layer):
      return []

    # Configure how to quantize activations.
    def get_activations_and_quantizers(self, layer):
      return []
    # Configure how to quantize outputs (may be equivalent to activations).
    def get_output_quantizers(self, layer):
      return []

    def get_config(self):
      return {}

def quant_annotated_cnn(trackShape, args):
    #filter1=30, filter2=10, kernel1=50, kernel2=15
    inputFeatures = keras.layers.Input(trackShape)
    timeDist = quantize_annotate_layer(keras.layers.Dense(args.nHidden, activation = 'relu', input_shape = trackShape), DefaultDenseQuantizeConfig(args.quantBits))(inputFeatures)
    drop1 = keras.layers.Dropout(0.6)(timeDist)
    conv1 = quantize_annotate_layer(keras.layers.Conv1D(filters = args.filter1, kernel_size = args.kernel1, activation = "relu"), DefaultDenseQuantizeConfig(args.quantBits))(drop1)
    conv2 = quantize_annotate_layer(keras.layers.Conv1D(filters = args.filter2, kernel_size = args.kernel2, activation = "relu"), DefaultDenseQuantizeConfig(args.quantBits))(conv1)
    flatten = keras.layers.Flatten()(conv2)
    dense = quantize_annotate_layer(keras.layers.Dense(args.nHidden, activation = "relu"), DefaultDenseQuantizeConfig(args.quantBits))(flatten)
    drop2 = keras.layers.Dropout(0.6)(dense)
    out = quantize_annotate_layer(keras.layers.Dense(1), DefaultDenseQuantizeConfig(args.quantBits))(drop2)
    return keras.models.Model(inputs = inputFeatures, outputs = out)

def quant_annotated_cnn2(trackShape, args): 
    inputFeatures = keras.layers.Input(trackShape)
    timeDist = quantize_annotate_layer(keras.layers.Dense(args.nHidden, activation = 'relu', input_shape = trackShape), DefaultDenseQuantizeConfig(args.quantBits))(inputFeatures)
    drop1 = quantize_annotate_layer(keras.layers.Dropout(0.6), DefaultDenseQuantizeConfig(args.quantBits))(timeDist)
    conv1 = quantize_annotate_layer(keras.layers.Conv1D(filters = args.filter1, kernel_size = args.kernel1, activation = "relu"), DefaultDenseQuantizeConfig(args.quantBits))(drop1)
    conv2 = quantize_annotate_layer(keras.layers.Conv1D(filters = args.filter2, kernel_size = args.kernel2, activation = "relu"), DefaultDenseQuantizeConfig(args.quantBits))(conv1)
    flatten = quantize_annotate_layer(keras.layers.Flatten(), DefaultDenseQuantizeConfig(args.quantBits))(conv2)
    dense = quantize_annotate_layer(keras.layers.Dense(args.nHidden, activation = "relu"), DefaultDenseQuantizeConfig(args.quantBits))(flatten)
    drop2 = quantize_annotate_layer(keras.layers.Dropout(0.6), DefaultDenseQuantizeConfig(args.quantBits))(dense)
    out = quantize_annotate_layer(keras.layers.Dense(1), DefaultDenseQuantizeConfig(args.quantBits))(drop2)
    return keras.models.Model(inputs = inputFeatures, outputs = out)

def quant_annotated_cnn_no_dropout(trackShape, args): 
    inputFeatures = keras.layers.Input(trackShape)
    timeDist = quantize_annotate_layer(keras.layers.Dense(args.nHidden, activation = 'relu', input_shape = trackShape), DefaultDenseQuantizeConfig(args.quantBits))(inputFeatures)
    conv1 = quantize_annotate_layer(keras.layers.Conv1D(filters = args.filter1, kernel_size = args.kernel1, activation = "relu"), DefaultDenseQuantizeConfig(args.quantBits))(timeDist)
    conv2 = quantize_annotate_layer(keras.layers.Conv1D(filters = args.filter2, kernel_size = args.kernel2, activation = "relu"), DefaultDenseQuantizeConfig(args.quantBits))(conv1)
    flatten = keras.layers.Flatten()(conv2)
    dense = quantize_annotate_layer(keras.layers.Dense(args.nHidden, activation = "relu"), DefaultDenseQuantizeConfig(args.quantBits))(flatten)
    out = quantize_annotate_layer(keras.layers.Dense(1), DefaultDenseQuantizeConfig(args.quantBits))(dense)
    return keras.models.Model(inputs = inputFeatures, outputs = out)


def quant_annotated_cnn_deep_conv(trackShape, args): 
    inputFeatures = keras.layers.Input(trackShape)
    timeDist = quantize_annotate_layer(keras.layers.Dense(args.nHidden, activation = 'relu', input_shape = trackShape), DefaultDenseQuantizeConfig(args.quantBits))(inputFeatures)
    drop1 = keras.layers.Dropout(0.6)(timeDist)
    conv1 = quantize_annotate_layer(keras.layers.Conv1D(filters = 30, kernel_size = 25, activation = "relu"), DefaultDenseQuantizeConfig(args.quantBits))(drop1)
    conv2 = quantize_annotate_layer(keras.layers.Conv1D(filters = 20, kernel_size = 10, activation = "relu"), DefaultDenseQuantizeConfig(args.quantBits))(conv1)
    conv3 = quantize_annotate_layer(keras.layers.Conv1D(filters = 10, kernel_size = 5, activation = "relu"), DefaultDenseQuantizeConfig(args.quantBits))(conv2)
    conv4 = quantize_annotate_layer(keras.layers.Conv1D(filters = 5, kernel_size = 3, activation = "relu"), DefaultDenseQuantizeConfig(args.quantBits))(conv3)
    flatten = keras.layers.Flatten()(conv4)
    
    dense = quantize_annotate_layer(keras.layers.Dense(args.nHidden, activation = "relu"), DefaultDenseQuantizeConfig(args.quantBits))(flatten)
    drop2 = keras.layers.Dropout(0.6)(dense)
    out = quantize_annotate_layer(keras.layers.Dense(1), DefaultDenseQuantizeConfig(args.quantBits))(drop2)
    return keras.models.Model(inputs = inputFeatures, outputs = out)

def quant_annotated_cnn_deep_conv_avg(trackShape, args): 
    inputFeatures = keras.layers.Input(trackShape)
    timeDist = quantize_annotate_layer(keras.layers.Dense(args.nHidden, activation = 'relu', input_shape = trackShape), DefaultDenseQuantizeConfig(args.quantBits))(inputFeatures)
    drop1 = keras.layers.Dropout(0.6)(timeDist)
    conv1 = quantize_annotate_layer(keras.layers.Conv1D(filters = 30, kernel_size = 25, activation = "relu"), DefaultDenseQuantizeConfig(args.quantBits))(drop1)
    conv2 = quantize_annotate_layer(keras.layers.Conv1D(filters = 20, kernel_size = 10, activation = "relu"), DefaultDenseQuantizeConfig(args.quantBits))(conv1)
    conv3 = quantize_annotate_layer(keras.layers.Conv1D(filters = 10, kernel_size = 5, activation = "relu"), DefaultDenseQuantizeConfig(args.quantBits))(conv2)
    conv4 = quantize_annotate_layer(keras.layers.Conv1D(filters = 5, kernel_size = 3, activation = "relu"), DefaultDenseQuantizeConfig(args.quantBits))(conv3)
    flatten = keras.layers.GlobalAveragePooling1D()(conv4)
    dense = quantize_annotate_layer(keras.layers.Dense(args.nHidden, activation = "relu"), DefaultDenseQuantizeConfig(args.quantBits))(flatten)
    drop2 = keras.layers.Dropout(0.6)(dense)
    out = quantize_annotate_layer(keras.layers.Dense(1), DefaultDenseQuantizeConfig(args.quantBits))(drop2)
    return keras.models.Model(inputs = inputFeatures, outputs = out)

def quant_annotated_cnn_parallel(trackShape, args):
    inputFeatures = keras.layers.Input(trackShape)
    #instead of embedding layers, to process the features before convolution and attention
    query = keras.layers.Dense(args.nHidden, activation = "relu", input_shape = trackShape)(inputFeatures)
    value = keras.layers.Dense(args.nHidden, activation = "relu", input_shape = trackShape)(inputFeatures)
    #queryC = keras.layers.Conv1D(filters = 25, kernel_size = 9, activation = "relu")(query)
    #valueC = keras.layers.Conv1D(filters = 25, kernel_size = 9, activation = "relu")(value)
    queryC = keras.layers.Conv1D(filters = args.filter1, kernel_size = args.kernel1, activation = "relu")(query)
    valueC = keras.layers.Conv1D(filters = args.filter2, kernel_size = args.kernel2, activation = "relu")(value)
    queryFlat = keras.layers.GlobalAveragePooling1D()(query)
    queryCFlat = keras.layers.GlobalAveragePooling1D()(queryC)
    concatenated = keras.layers.Concatenate()([queryFlat, queryCFlat])
    dense1 = keras.layers.Dense(args.nHidden, activation = "relu")(concatenated)
    dropout = keras.layers.Dropout(0.5)(dense1)
    out = keras.layers.Dense(1)(dropout)
    return keras.models.Model(inputs = inputFeatures, outputs = out)

def quant_annotated_cnn_parallel_fast(trackShape, args):
    inputFeatures = keras.layers.Input(trackShape)
    # --kernel1 15 --kernel2 15 --filter1 20 --filter2 20
    #sequence dist dense layer instead of embedding layer, to process the features before convolution and attention
    query = keras.layers.Dense(args.nHidden, activation = "relu", input_shape = trackShape)(inputFeatures)
    queryC = keras.layers.Conv1D(filters = args.filter1, kernel_size = args.kernel1, activation = "relu")(query)
    valueC = keras.layers.Conv1D(filters = args.filter2, kernel_size = args.kernel2, activation = "relu")(query)
    queryFlat = keras.layers.GlobalAveragePooling1D()(query)
    queryCFlat = keras.layers.GlobalAveragePooling1D()(queryC)
    concatenated = keras.layers.Concatenate()([queryFlat, queryCFlat])
    dense1 = keras.layers.Dense(args.nHidden, activation = "relu")(concatenated)
    dropout = keras.layers.Dropout(0.5)(dense1)
    out = keras.layers.Dense(1)(dropout)
    return keras.models.Model(inputs = inputFeatures, outputs = out)

def quant_annotated_cnn_L_attention_correct(trackShape, args):
    inputFeatures = keras.layers.Input(trackShape)
    # --kernel1 15 --kernel2 15 --filter1 20 --filter2 20
    #sequence dist dense layer instead of embedding layer, to process the features before convolution and attention
    query = keras.layers.Dense(args.nHidden, activation = "relu", input_shape = trackShape)(inputFeatures)
    value = keras.layers.Dense(args.nHidden, activation = "relu", input_shape = trackShape)(inputFeatures)
    #queryC = keras.layers.Conv1D(filters = 25, kernel_size = 9, activation = "relu")(query)
    #valueC = keras.layers.Conv1D(filters = 25, kernel_size = 9, activation = "relu")(value)
    queryC = keras.layers.Conv1D(filters = args.filter1, kernel_size = args.kernel1, activation = "relu")(query)
    valueC = keras.layers.Conv1D(filters = args.filter2, kernel_size = args.kernel2, activation = "relu")(value)
    attention = keras.layers.Attention(dropout = 0.5)([queryC, valueC])
    queryFlat = keras.layers.GlobalAveragePooling1D()(attention)
    queryCFlat = keras.layers.GlobalAveragePooling1D()(queryC)
    concatenated = keras.layers.Concatenate()([queryFlat, queryCFlat])
    dense1 = keras.layers.Dense(args.nHidden, activation = "relu")(concatenated)
    dropout = keras.layers.Dropout(0.5)(dense1)
    out = keras.layers.Dense(1)(dropout)
    return keras.models.Model(inputs = inputFeatures, outputs = out)

def quant_annotated_cnn_L_attention_correct_fast(trackShape, args):
    inputFeatures = keras.layers.Input(trackShape)
    # --kernel1 15 --kernel2 15 --filter1 20 --filter2 20
    #sequence dist dense layer instead of embedding layer, to process the features before convolution and attention
    query = keras.layers.Dense(args.nHidden, activation = "relu", input_shape = trackShape)(inputFeatures)
    queryC = keras.layers.Conv1D(filters = args.filter1, kernel_size = args.kernel1, activation = "relu")(query)
    valueC = keras.layers.Conv1D(filters = args.filter2, kernel_size = args.kernel2, activation = "relu")(query)
    attention = keras.layers.Attention(dropout = 0.5)([queryC, valueC])
    queryFlat = keras.layers.GlobalAveragePooling1D()(attention)
    queryCFlat = keras.layers.GlobalAveragePooling1D()(queryC)
    concatenated = keras.layers.Concatenate()([queryFlat, queryCFlat])
    dense1 = keras.layers.Dense(args.nHidden, activation = "relu")(concatenated)
    dropout = keras.layers.Dropout(0.5)(dense1)
    out = keras.layers.Dense(1)(dropout)
    return keras.models.Model(inputs = inputFeatures, outputs = out)

if __name__ == '__main__':

    nBatch = 37
    nTracks = 100
    nFeatures = 18

    nHidden = 8

    inputFeatures = keras.layers.Input((nTracks, nFeatures))
    inputFeaturesMasked = keras.layers.Masking(mask_value = -999, name = 'maskFeatures')(inputFeatures)

    tracks, hidden = keras.layers.GRU(nHidden, activation = 'relu', return_state = True, return_sequences = True)(inputFeaturesMasked)

    context, attention = AttentionBahdanau(nHidden)([hidden, tracks])

    dense = keras.layers.Dense(nHidden, activation = 'relu')(context)
    out = keras.layers.Dense(1, activation = 'sigmoid')(dense)

    model = keras.models.Model(inputs = [inputFeatures], outputs = out)
    model.summary(line_length = 150)

    inputs = np.random.normal(0, 1, size = (nBatch, nTracks, nFeatures))

    model.predict(inputs)
