from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
import tensorflow.keras.layers as Klayers
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.quantization.keras import quantize_config

### --- Not sure how this works yet, but solves problems with quantising RNNs --- ###
### forgive american spelling, want it to be consitent ###

class quantizeRNN(quantize_config.QuantizeConfig):
    def get_weights_and_quantizers(self, layer):
        return [(layer.kernel, LastValueQuantizer())]
    def get_activations_and_quantizers(self, layer):
        return [(layer.activation, MovingAverageQuantizer())]
    def set_quantize_weights(self, layer, quantize_weights):
        layer.kernel = quantize_weights[0]
    def set_quantize_activations(self, layer, quantize_activations):
        layer.activation = quantize_activations[0]
    def get_output_quantizers(self, layer):
        # Does not quantize output, since we return an empty list.
        return []
    def get_config(self):
        return {}

def tagNetworkSmallDropoutQuant(trackShape, units):

    '''

    Tag classifier taking input shape (nBatch, nTracks, nFeatures) with masking.

    Reduced number of layers and parameters, for testing.

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    input_1 (InputLayer)         (None, 100, 18)           0
    _________________________________________________________________
    mask (Masking)               (None, 100, 18)           0
    _________________________________________________________________
    td_dense1 (TimeDistributed)  (None, 100, 16)           304
    _________________________________________________________________
    noseq_gru (GRU)              (None, 16)                1584
    _________________________________________________________________
    out_dense_1 (Dense)          (None, 16)                272
    _________________________________________________________________
    outputTag (Dense)            (None, 1)                 17
    =================================================================
    Total params: 2,177
    Trainable params: 2,177
    Non-trainable params: 0
    _________________________________________________________________
    '''
    quantize_config = quantizeRNN()
    trackInput = Klayers.Input(trackShape)
    tracks = Klayers.Masking(mask_value = -999, name = 'mask')(trackInput)

    tracks = Klayers.TimeDistributed(Klayers.Dense(units, activation = 'relu'), name = 'td_dense1')(tracks)
    """
    tracks = tfmot.quantization.keras.quantize_annotate_layer( Klayers.GRU(units, activation = 'relu', return_sequences = False, reset_after = True, recurrent_dropout = 0.5, name = 'noseq_gru'), quantize_config = quantize_config)(tracks)
    """


    tracks = Klayers.GRU(units, activation = 'relu', return_sequences = False, reset_after = True, recurrent_dropout = 0.5, name = 'noseq_gru')(tracks)
    tracks = tfmot.quantization.keras.quantize_annotate_layer(Klayers.Dense(units, activation = 'relu', name = 'td_dense2'))(tracks)

    tracks = Klayers.Dropout(rate = 0.5)(tracks)

    outputTag = Klayers.Dense(units = 1, activation='sigmoid', name = 'outputTag')(tracks)

    return Model(inputs = trackInput, outputs = outputTag)
