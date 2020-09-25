""" Functional definitions for attention based RNNs, along with custom self attention layers """

import numpy as np

import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow import dtypes, matmul, Tensor, tensordot, transpose

class AttentionBahdanau(keras.layers.Layer):
    def __init__(self, units):

        self.units = units

        super(AttentionBahdanau, self).__init__()

    def build(self, input_shape):

        # Need to know input shapes in order to register
        # number of parameters (weights)

        # These undergo transformations in call(), so can't
        # just build these here and request input_shape
        # and output_shape

        # https://www.tensorflow.org/tutorials/text/nmt_with_attention

        self.W1 = keras.layers.Dense(self.units, activation = 'relu', use_bias = True)
        self.W2 = keras.layers.Dense(self.units, activation = 'relu', use_bias = True)
        self.V = keras.layers.Dense(1, activation = None, use_bias = True)

        super(AttentionBahdanau, self).build(input_shape)

    def compute_output_shape(self, input_shape):

        # Neccessary - automatic inference, gets the shape wrong!

        shape_hidden, shape_values = input_shape
        batch_size = shape_hidden[0]

        shape_context = (batch_size, shape_hidden[1])
        shape_attention_weights = (batch_size, shape_values[1], 1)

        return [shape_context, shape_attention_weights]

    def compute_mask(self, inputs, mask = None):
        return mask

    def count_params(self):
        return 2 * (self.units * self.units)+ 3 * self.units + 1

    def call(self, input):

        query, values = input

        # Query: (1, batch, hidden)
        # Values: (batch, seq_len, hidden)

        # -> (batch, 1, hidden)
        # To perform addition to calculate the score
        hidden_with_time_axis = K.expand_dims(query, 1)
        
        # score shape == (batch_size, max_length, hidden_size)
        score = self.V(K.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))
        # attention_weights -> (batch_size, max_length, 1)
        attention_weights = K.softmax(score, axis=1)

        # context_vector shape after sum -? (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = K.sum(context_vector, axis = 1)

        return [context_vector, attention_weights]

    def get_config(self):

        config = super().get_config().copy()
        config.update({
           'units': self.units,
        })
        return config

class AttentionBahdanauSimple(keras.layers.Layer):
    def __init__(self, units):

        self.units = units

        super(AttentionBahdanauSimple, self).__init__()

    def build(self, input_shape):

        # Need to know input shapes in order to register
        # number of parameters (weights)

        # These undergo transformations in call(), so can't
        # just build these here and request input_shape
        # and output_shape

        # https://www.tensorflow.org/tutorials/text/nmt_with_attention

        self.W1 = keras.layers.Dense(self.units, activation = None, use_bias = False)
        self.W2 = keras.layers.Dense(self.units, activation = None,  use_bias = False)
        self.V = keras.layers.Dense(1, activation = None, use_bias = False)

        super(AttentionBahdanauSimple, self).build(input_shape)

    def compute_output_shape(self, input_shape):

        # Neccessary - automatic inference, gets the shape wrong!

        shape_hidden, shape_values = input_shape
        batch_size = shape_hidden[0]

        shape_context = (batch_size, shape_hidden[1])
        shape_attention_weights = (batch_size, shape_values[1], 1)

        return [shape_context, shape_attention_weights]

    def compute_mask(self, inputs, mask = None):
        return mask

    def count_params(self):
        return 2 * (self.units * self.units)+ self.units 

    def call(self, input):

        query, values = input
        

        # Query: (1, batch, hidden)
        # Values: (batch, seq_len, hidden)

        # -> (batch, 1, hidden)
        # To perform addition to calculate the score
        hidden_with_time_axis = K.expand_dims(query, 1)
        
        # score shape == (batch_size, max_length, hidden_size)
        score = self.V(K.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))        
        # attention_weights -> (batch_size, max_length, 1)
        attention_weights = K.softmax(score, axis=1)

        # context_vector shape after sum -? (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = K.sum(context_vector, axis = 1)

        return [context_vector, attention_weights]

    def get_config(self):

        config = super().get_config().copy()
        config.update({
           'units': self.units,
        })
        return config

    
class AttentionLuong(keras.layers.Layer):
    def __init__(self, units):
        
        self.units = units
        super(AttentionLuong, self).__init__()

    def build(self, input_shape):
        # Need to know input shapes in order to register
        # number of parameters (weights)

        # These undergo transformations in call(), so can't
        # just build these here and request input_shape
        # and output_shape

        # https://www.tensorflow.org/tutorials/text/nmt_with_attention

        self.Wa = keras.layers.Dense(self.units, activation = 'relu', use_bias = True)
        super(AttentionLuong, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        # Neccessary - automatic inference, gets the shape wrong!

        shape_hidden, shape_values = input_shape
        batch_size = shape_hidden[0]

        shape_context = (batch_size, shape_hidden[1])
        shape_attention_weights = (batch_size, shape_values[1], 1)

        return [shape_context, shape_attention_weights]

    def compute_mask(self, inputs, mask = None):
        return mask
    
    #def count_params(self):
    #    return 2 * (self.units * self.units)+ 3 * self.units + 1
    
    def call(self, input):
        query, values = input

        # Query: (1, batch, hidden)
        # Values: (batch, seq_len, hidden)

        # -> (batch, 1, hidden)
        # To perform addition to calculate the score
        hidden_with_time_axis = K.expand_dims(query, 1)
        # score shape == (batch_size, max_length, hidden_size)
        
        
        score = matmul(values, self.Wa(hidden_with_time_axis), transpose_b = True)
        # attention_weights -> (batch_size, max_length, 1)
        attention_weights = K.softmax(score, axis=1)
        
        # context_vector shape after sum -? (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = K.sum(context_vector, axis = 1)
        print(attention_weights)
        return [context_vector, attention_weights]

    def get_config(self):

        config = super().get_config().copy()
        config.update({
           'units': self.units,
        })
        return config

class AttentionLuongSimple(keras.layers.Layer):
    def __init__(self, units):
        
        self.units = units
        super(AttentionLuongSimple, self).__init__()

    def build(self, input_shape):
        # Need to know input shapes in order to register
        # number of parameters (weights)

        # These undergo transformations in call(), so can't
        # just build these here and request input_shape
        # and output_shape

        # https://www.tensorflow.org/tutorials/text/nmt_with_attention

        super(AttentionLuongSimple, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        # Neccessary - automatic inference, gets the shape wrong!

        shape_hidden, shape_values = input_shape
        batch_size = shape_hidden[0]

        shape_context = (batch_size, shape_hidden[1])
        shape_attention_weights = (batch_size, shape_values[1], 1)

        return [shape_context, shape_attention_weights]

    def compute_mask(self, inputs, mask = None):
        return mask
    
    def call(self, input):
        query, values = input

        # Query: (1, batch, hidden)
        # Values: (batch, seq_len, hidden)

        # -> (batch, 1, hidden)
        # To perform addition to calculate the score
        hidden_with_time_axis = K.expand_dims(query, 1)
        # score shape == (batch_size, max_length, hidden_size)
        
        
        score = matmul(values, hidden_with_time_axis, transpose_b = True)
        # attention_weights -> (batch_size, max_length, 1)
        attention_weights = K.softmax(score, axis=1)
        
        # context_vector shape after sum -? (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = K.sum(context_vector, axis = 1)
        print(attention_weights)
        return [context_vector, attention_weights]

    def get_config(self):

        config = super().get_config().copy()
        config.update({
           'units': self.units,
        })
        return config


def attentionNetwork(trackShape, nHidden):
    inputFeatures = keras.layers.Input(trackShape)
    inputFeaturesMasked = keras.layers.Masking(mask_value = -999, name = 'maskFeatures')(inputFeatures)
    
    tracks, hidden = keras.layers.GRU(nHidden, activation = 'relu', return_state = True, return_sequences = True)(inputFeaturesMasked)
    
    context, attention = AttentionBahdanau(nHidden)([hidden, tracks])
    print(tf.make_ndarray(attention))
    dense = keras.layers.Dense(nHidden, activation = 'relu')(context)
    out = keras.layers.Dense(1, activation = 'sigmoid')(dense)

    return keras.models.Model(inputs = inputFeatures, outputs = out)

def attentionNetworkDeep(trackShape, nHidden):
    inputFeatures = keras.layers.Input(trackShape)
    inputFeaturesMasked = keras.layers.Masking(mask_value = -999, name = 'maskFeatures')(inputFeatures)
    timeDist = Klayers.TimeDistributed(Klayers.Dense(nHidden, activation = 'relu'), name = 'td_dense1')(inputFeaturesMasked)

    tracks, hidden = keras.layers.GRU(nHidden, activation = 'relu', return_state = True, return_sequences = True)(timeDist)
    
    context, attention = AttentionBahdanau(nHidden)([hidden, tracks])
    print(tf.make_ndarray(attention))
    dense = keras.layers.Dense(nHidden, activation = 'relu')(context)
    out = keras.layers.Dense(1, activation = 'sigmoid')(dense)

    return keras.models.Model(inputs = inputFeatures, outputs = out)

def attentionNetworkSimple(trackShape, nHidden):
    inputFeatures = keras.layers.Input(trackShape)
    inputFeaturesMasked = keras.layers.Masking(mask_value = -999, name = 'maskFeatures')(inputFeatures)

    tracks, hidden = keras.layers.GRU(nHidden, activation = 'relu', return_state = True, return_sequences = True)(inputFeaturesMasked)

    context, attention = AttentionBahdanauSimple(nHidden)([hidden, tracks])
    
    dense = keras.layers.Dense(nHidden, activation = 'relu')(context)
    out = keras.layers.Dense(1, activation = 'sigmoid')(dense)

    return keras.models.Model(inputs = inputFeatures, outputs = out)

def attentionNetworkLuong(trackShape, nHidden):
    inputFeatures = keras.layers.Input(trackShape)
    inputFeaturesMasked = keras.layers.Masking(mask_value = -999, name = 'maskFeatures')(inputFeatures)

    tracks, hidden = keras.layers.GRU(nHidden, activation = 'relu', return_state = True, return_sequences = True)(inputFeaturesMasked)

    context, attention = AttentionLuong(nHidden)([hidden, tracks])

    dense = keras.layers.Dense(nHidden, activation = 'relu')(context)
    out = keras.layers.Dense(1, activation = 'sigmoid')(dense)

    return keras.models.Model(inputs = inputFeatures, outputs = out)

def attentionNetworkLuongSimple(trackShape, nHidden):
    inputFeatures = keras.layers.Input(trackShape)
    inputFeaturesMasked = keras.layers.Masking(mask_value = -999, name = 'maskFeatures')(inputFeatures)

    tracks, hidden = keras.layers.GRU(nHidden, activation = 'relu', return_state = True, return_sequences = True)(inputFeaturesMasked)

    context, attention = AttentionLuongSimple(nHidden)([hidden, tracks])

    dense = keras.layers.Dense(nHidden, activation = 'relu')(context)
    out = keras.layers.Dense(1, activation = 'sigmoid')(dense)

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
