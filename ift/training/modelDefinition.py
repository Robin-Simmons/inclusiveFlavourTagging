from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
import tensorflow.keras.layers as Klayers
from tensorflow.keras import regularizers

def catNetwork(trackShape, trackCategories):

    '''

    Track category classifier taking input with the same shape as the tag network, using a recurrent layer.
    Outputs are returned per event as shape (nBatch, nTracks, nCategories).

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    input_1 (InputLayer)         (None, 100, 18)           0
    _________________________________________________________________
    mask (Masking)               (None, 100, 18)           0
    _________________________________________________________________
    td_dense1 (TimeDistributed)  (None, 100, 32)           608
    _________________________________________________________________
    track_gru (GRU)              (None, 100, 32)           6240
    _________________________________________________________________
    noseq_gru (GRU)              (None, 100, 32)           6240
    _________________________________________________________________
    time_distributed_1 (TimeDist (None, 100, 32)           1056
    _________________________________________________________________
    time_distributed_2 (TimeDist (None, 100, 32)           1056
    _________________________________________________________________
    outputCat (Dense)            (None, 100, 4)            132
    =================================================================
    Total params: 15,332
    Trainable params: 15,332
    Non-trainable params: 0
    _________________________________________________________________

    '''

    trackInput =Klayers.Input(trackShape)
    tracks =Klayers.Masking(mask_value = -999, name = 'mask')(trackInput)

    tracks =Klayers.TimeDistributed(Klayers.Dense(32, activation = 'relu'), name = 'td_dense1')(tracks)

    tracks =Klayers.GRU(32, activation = 'relu', return_sequences = True, name = 'track_gru')(tracks)
    tracks =Klayers.GRU(32, activation = 'relu', return_sequences = True, recurrent_dropout = 0.5, name = 'noseq_gru')(tracks)

    tracks =Klayers.TimeDistributed(Klayers.Dense(32, activation = 'relu', name = 'out_dense_1'))(tracks)
    tracks =Klayers.TimeDistributed(Klayers.Dense(32, activation = 'relu', name = 'out_dense_4'))(tracks)

    outputCat =Klayers.Dense(trackCategories, activation = 'softmax', name = 'outputCat')(tracks)

    return Model(inputs = trackInput, outputs = outputCat)

def catNetworkFlat(trackShape, trackCategories):

    '''

    Track category classifier taking input flattened across tracks, shape (nBatch * nTracks', nFeatures),
    where nTracks' are the total number of non-zero entries.

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    input_1 (InputLayer)         (None, 18)                0
    _________________________________________________________________
    cat_dense1 (Dense)           (None, 32)                608
    _________________________________________________________________
    cat_dense2 (Dense)           (None, 32)                1056
    _________________________________________________________________
    batch_normalization_1 (Batch (None, 32)                128
    _________________________________________________________________
    cat_dense3 (Dense)           (None, 32)                1056
    _________________________________________________________________
    cat_dense4 (Dense)           (None, 32)                1056
    _________________________________________________________________
    batch_normalization_2 (Batch (None, 32)                128
    _________________________________________________________________
    cat_dense5 (Dense)           (None, 32)                1056
    _________________________________________________________________
    cat_dense6 (Dense)           (None, 32)                1056
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 32)                0
    _________________________________________________________________
    outputCat (Dense)            (None, 4)                 132
    =================================================================
    Total params: 6,276
    Trainable params: 6,148
    Non-trainable params: 128
    _________________________________________________________________

    '''

    trackInput =Klayers.Input(trackShape)

    tracks =Klayers.Dense(32, activation = 'relu', name = 'cat_dense1')(trackInput)
    tracks =Klayers.Dense(32, activation = 'relu', name = 'cat_dense2')(tracks)

    tracks =Klayers.BatchNormalization(momentum = 0.99)(tracks)
    tracks =Klayers.Dense(32, activation = 'relu', name = 'cat_dense3')(tracks)
    tracks =Klayers.Dense(32, activation = 'relu', name = 'cat_dense4')(tracks)

    tracks =Klayers.BatchNormalization(momentum = 0.99)(tracks)
    tracks =Klayers.Dense(32, activation = 'relu', name = 'cat_dense5')(tracks)
    tracks =Klayers.Dense(32, activation = 'relu', name = 'cat_dense6')(tracks)
    tracks =Klayers.Dropout(0.5)(tracks)

    outputCat =Klayers.Dense(trackCategories, activation = 'softmax', name = 'outputCat')(tracks)

    return Model(inputs = trackInput, outputs = outputCat)

def tagNetworkSmall(trackShape, units):

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

    trackInput =Klayers.Input(trackShape)
    tracks =Klayers.Masking(mask_value = -999, name = 'mask')(trackInput)

    tracks =Klayers.TimeDistributed(Klayers.Dense(units, activation = 'relu'), name = 'td_dense1')(tracks)

    tracks =Klayers.GRU(units, activation = 'relu', reset_after = True, return_sequences = False,  name = 'noseq_gru')(tracks)
    #recurrent_dropout = 0.5
    tracks =Klayers.Dense(units, activation = 'relu', name = 'out_dense_1')(tracks)

    outputTag =Klayers.Dense(1, activation='sigmoid', name = 'outputTag')(tracks)

    return Model(inputs = trackInput, outputs = outputTag)

def tagNetworkSmallNoTD(trackShape, units):

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

    trackInput =Klayers.Input(trackShape)
    tracks =Klayers.Masking(mask_value = -999, name = 'mask')(trackInput)

    tracks =Klayers.GRU(units, activation = 'relu', reset_after = True, return_sequences = False,  name = 'noseq_gru')(tracks)
    #recurrent_dropout = 0.5
    tracks =Klayers.Dense(units, activation = 'relu', name = 'out_dense_1')(tracks)

    outputTag =Klayers.Dense(1, activation='sigmoid', name = 'outputTag')(tracks)

    return Model(inputs = trackInput, outputs = outputTag)

def tagNetworkSmallL2Reg(trackShape):

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

    trackInput =Klayers.Input(trackShape, units)
    tracks = Klayers.Masking(mask_value = -999, name = 'mask')(trackInput)

    tracks = Klayers.TimeDistributed(Klayers.Dense(16, activation = 'relu'), name = 'td_dense1')(tracks)

    tracks = Klayers.GRU(16, activation = 'relu', return_sequences = False, recurrent_dropout = 0.5, name = 'noseq_gru')(tracks)

    tracks = Klayers.Dense(16, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.l2(1e-4),
    activity_regularizer=regularizers.l2(1e-5), activation = 'relu', name = 'out_dense_1')(tracks)

    outputTag =Klayers.Dense(1, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.l2(1e-4),
    activity_regularizer=regularizers.l2(1e-5), activation='sigmoid', name = 'outputTag')(tracks)

    return Model(inputs = trackInput, outputs = outputTag)

def tagNetworkSmallDropout(trackShape, units):

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

    trackInput =Klayers.Input(trackShape)
    tracks =Klayers.Masking(mask_value = -999, name = 'mask')(trackInput)

    tracks =Klayers.Dense(units, input_shape = trackShape,  activation = 'relu')(tracks)

    tracks =Klayers.GRU(units, activation = 'relu', reset_after = True, return_sequences = False, recurrent_dropout = 0.7, name = 'noseq_gru')(tracks)
    
    tracks =Klayers.Dense(units, activation = 'relu', name = 'td_dense2')(tracks)

    tracks =Klayers.Dropout(rate = 0.7)(tracks)

    outputTag =Klayers.Dense(units = 1, activation='sigmoid', name = 'outputTag')(tracks)

    return Model(inputs = trackInput, outputs = outputTag)

def tagNetworkMediumSmallSmall(trackShape, units):

    '''

    Tag classifier taking input shape (nBatch, nTracks, nFeatures) with masking.

    Reduced number of layers and parameters, for testing.

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         [(None, 100, 18)]         0         
    _________________________________________________________________
    mask (Masking)               (None, 100, 18)           0         
    _________________________________________________________________
    td_dense1 (TimeDistributed)  (None, 100, 32)           608       
    _________________________________________________________________
    noseq_gru (GRU)              (None, 16)                2400      
    _________________________________________________________________
    out_dense_1 (Dense)          (None, 16)                272       
    _________________________________________________________________
    outputTag (Dense)            (None, 1)                 17        
    =================================================================
    Total params: 3,297
    Trainable params: 3,297
    Non-trainable params: 0
    _________________________________________________________________
    '''

    trackInput =Klayers.Input(trackShape)
    tracks =Klayers.Masking(mask_value = -999, name = 'mask')(trackInput)

    tracks =Klayers.TimeDistributed(Klayers.Dense(32, activation = 'relu'), name = 'td_dense1')(tracks)

    tracks =Klayers.GRU(16, activation = 'relu', return_sequences = False, recurrent_dropout = 0.5, name = 'noseq_gru')(tracks)

    tracks =Klayers.Dense(16, activation = 'relu', name = 'out_dense_1')(tracks)

    outputTag =Klayers.Dense(1, activation='sigmoid', name = 'outputTag')(tracks)

    return Model(inputs = trackInput, outputs = outputTag)

def tagNetworkMediumSmall(trackShape, units):

    '''

    Tag classifier taking input shape (nBatch, nTracks, nFeatures) with masking.

    Reduced number of layers and parameters, for testing.

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         [(None, 100, 18)]         0         
    _________________________________________________________________
    mask (Masking)               (None, 100, 18)           0         
    _________________________________________________________________
    td_dense1 (TimeDistributed)  (None, 100, 32)           608       
    _________________________________________________________________
    noseq_gru (GRU)              (None, 32)                6336      
    _________________________________________________________________
    out_dense_1 (Dense)          (None, 16)                528       
    _________________________________________________________________
    outputTag (Dense)            (None, 1)                 17        
    =================================================================
    Total params: 7,489
    Trainable params: 7,489
    Non-trainable params: 0
    _________________________________________________________________
    
    '''

    trackInput =Klayers.Input(trackShape)
    tracks =Klayers.Masking(mask_value = -999, name = 'mask')(trackInput)

    tracks =Klayers.TimeDistributed(Klayers.Dense(32, activation = 'relu'), name = 'td_dense1')(tracks)

    tracks =Klayers.GRU(32, activation = 'relu', return_sequences = False, recurrent_dropout = 0.5, name = 'noseq_gru')(tracks)

    tracks =Klayers.Dense(16, activation = 'relu', name = 'out_dense_1')(tracks)

    outputTag =Klayers.Dense(1, activation='sigmoid', name = 'outputTag')(tracks)

    return Model(inputs = trackInput, outputs = outputTag)

def tagNetworkMedium(trackShape, units):

    '''

    Tag classifier taking input shape (nBatch, nTracks, nFeatures) with masking.

    Reduced number of layers and parameters, for testing.

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         [(None, 100, 18)]         0         
    _________________________________________________________________
    mask (Masking)               (None, 100, 18)           0         
    _________________________________________________________________
    td_dense1 (TimeDistributed)  (None, 100, 32)           608       
    _________________________________________________________________
    noseq_gru (GRU)              (None, 32)                6336      
    _________________________________________________________________
    out_dense_1 (Dense)          (None, 32)                1056      
    _________________________________________________________________
    outputTag (Dense)            (None, 1)                 33        
    =================================================================
    Total params: 8,033
    Trainable params: 8,033
    Non-trainable params: 0
    _________________________________________________________________
    '''

    trackInput =Klayers.Input(trackShape)
    tracks =Klayers.Masking(mask_value = -999, name = 'mask')(trackInput)

    tracks =Klayers.TimeDistributed(Klayers.Dense(32, activation = 'relu'), name = 'td_dense1')(tracks)

    tracks =Klayers.GRU(32, activation = 'relu', return_sequences = False, recurrent_dropout = 0.5, name = 'noseq_gru')(tracks)

    tracks =Klayers.Dense(32, activation = 'relu', name = 'out_dense_1')(tracks)

    outputTag =Klayers.Dense(1, activation='sigmoid', name = 'outputTag')(tracks)

    return Model(inputs = trackInput, outputs = outputTag)

def testNetwork(trackShape, units):
    trackInput =Klayers.Input(trackShape)
    tracks =Klayers.Masking(mask_value = -999, name = 'mask')(trackInput)

    tracks = Klayers.Flatten()(tracks)

    outputTag =Klayers.Dense(1, activation='sigmoid', name = 'outputTag')(tracks)

    return Model(inputs = trackInput, outputs = outputTag)


def tagNetwork(trackShape, units):

    '''

    Tag classifier taking input shape (nBatch, nTracks, nFeatures) with masking.

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    input_1 (InputLayer)         (None, 100, 18)           0
    _________________________________________________________________
    mask (Masking)               (None, 100, 18)           0
    _________________________________________________________________
    td_dense1 (TimeDistributed)  (None, 100, 32)           608
    _________________________________________________________________
    td_dense2 (TimeDistributed)  (None, 100, 32)           1056
    _________________________________________________________________
    track_gru (GRU)              (None, 100, 32)           6240
    _________________________________________________________________
    noseq_gru (GRU)              (None, 32)                6240
    _________________________________________________________________
    out_dense_1 (Dense)          (None, 32)                1056
    _________________________________________________________________
    out_dense_2 (Dense)          (None, 32)                1056
    _________________________________________________________________
    out_dense_3 (Dense)          (None, 32)                1056
    _________________________________________________________________
    outputTag (Dense)            (None, 1)                 33
    =================================================================
    Total params: 17,345
    Trainable params: 17,345
    Non-trainable params: 0
    _________________________________________________________________

    '''

    trackInput =Klayers.Input(trackShape)
    tracks =Klayers.Masking(mask_value = -999, name = 'mask')(trackInput)

    tracks =Klayers.TimeDistributed(Klayers.Dense(32, activation = 'relu'), name = 'td_dense1')(tracks)
    tracks =Klayers.TimeDistributed(Klayers.Dense(32, activation = 'relu'), name = 'td_dense2')(tracks)

    tracks =Klayers.GRU(32, activation = 'relu', reset_after = True, return_sequences = True, name = 'track_gru')(tracks)
    tracks =Klayers.GRU(32, activation = 'relu', reset_after = True, return_sequences = False, recurrent_dropout = 0.5, name = 'noseq_gru')(tracks)

    tracks =Klayers.Dense(32, activation = 'relu', name = 'out_dense_1')(tracks)
    tracks =Klayers.Dense(32, activation = 'relu', name = 'out_dense_2')(tracks)
    tracks =Klayers.Dense(32, activation = 'relu', name = 'out_dense_3')(tracks)

    outputTag =Klayers.Dense(1, activation='sigmoid', name = 'outputTag')(tracks)

    return Model(inputs = trackInput, outputs = outputTag)
def tagNetworkShallow(trackShape, units):

    '''

    Tag classifier taking input shape (nBatch, nTracks, nFeatures) with masking.

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    input_1 (InputLayer)         (None, 100, 18)           0
    _________________________________________________________________
    mask (Masking)               (None, 100, 18)           0
    _________________________________________________________________
    td_dense1 (TimeDistributed)  (None, 100, 32)           608
    _________________________________________________________________
    td_dense2 (TimeDistributed)  (None, 100, 32)           1056
    _________________________________________________________________
    track_gru (GRU)              (None, 100, 32)           6240
    _________________________________________________________________
    noseq_gru (GRU)              (None, 32)                6240
    _________________________________________________________________
    out_dense_1 (Dense)          (None, 32)                1056
    _________________________________________________________________
    out_dense_2 (Dense)          (None, 32)                1056
    _________________________________________________________________
    out_dense_3 (Dense)          (None, 32)                1056
    _________________________________________________________________
    outputTag (Dense)            (None, 1)                 33
    =================================================================
    Total params: 17,345
    Trainable params: 17,345
    Non-trainable params: 0
    _________________________________________________________________

    '''

    trackInput =Klayers.Input(trackShape)
    tracks =Klayers.Masking(mask_value = -999, name = 'mask')(trackInput)

    tracks =Klayers.TimeDistributed(Klayers.Dense(32, activation = 'relu'), name = 'td_dense1')(tracks)
    tracks =Klayers.TimeDistributed(Klayers.Dense(32, activation = 'relu'), name = 'td_dense2')(tracks)

    tracks =Klayers.GRU(32, activation = 'relu', return_sequences = True, name = 'track_gru')(tracks)
    tracks =Klayers.GRU(32, activation = 'relu', return_sequences = False, recurrent_dropout = 0.5, name = 'noseq_gru')(tracks)

    tracks =Klayers.Dense(88, activation = 'relu', name = 'out_dense_1')(tracks)

    outputTag =Klayers.Dense(1, activation='sigmoid', name = 'outputTag')(tracks)

    return Model(inputs = trackInput, outputs = outputTag)

def tagNetworkMediumDepth(trackShape, units):

    '''

    Tag classifier taking input shape (nBatch, nTracks, nFeatures) with masking.

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    input_1 (InputLayer)         (None, 100, 18)           0
    _________________________________________________________________
    mask (Masking)               (None, 100, 18)           0
    _________________________________________________________________
    td_dense1 (TimeDistributed)  (None, 100, 32)           608
    _________________________________________________________________
    td_dense2 (TimeDistributed)  (None, 100, 32)           1056
    _________________________________________________________________
    track_gru (GRU)              (None, 100, 32)           6240
    _________________________________________________________________
    noseq_gru (GRU)              (None, 32)                6240
    _________________________________________________________________
    out_dense_1 (Dense)          (None, 32)                1056
    _________________________________________________________________
    out_dense_2 (Dense)          (None, 32)                1056
    _________________________________________________________________
    out_dense_3 (Dense)          (None, 32)                1056
    _________________________________________________________________
    outputTag (Dense)            (None, 1)                 33
    =================================================================
    Total params: 17,345
    Trainable params: 17,345
    Non-trainable params: 0
    _________________________________________________________________

    '''

    trackInput =Klayers.Input(trackShape)
    tracks =Klayers.Masking(mask_value = -999, name = 'mask')(trackInput)

    tracks =Klayers.TimeDistributed(Klayers.Dense(32, activation = 'relu'), name = 'td_dense1')(tracks)
    tracks =Klayers.TimeDistributed(Klayers.Dense(32, activation = 'relu'), name = 'td_dense2')(tracks)

    tracks =Klayers.GRU(32, activation = 'relu', return_sequences = True, name = 'track_gru')(tracks)
    tracks =Klayers.GRU(32, activation = 'relu', return_sequences = False, recurrent_dropout = 0.5, name = 'noseq_gru')(tracks)

    tracks =Klayers.Dense(42, activation = 'relu', name = 'out_dense_1')(tracks)
    tracks =Klayers.Dense(42, activation = 'relu', name = 'out_dense_2')(tracks)
    outputTag =Klayers.Dense(1, activation='sigmoid', name = 'outputTag')(tracks)

    return Model(inputs = trackInput, outputs = outputTag)

def tagNetworkDeep(trackShape, units):

    '''

    Tag classifier taking input shape (nBatch, nTracks, nFeatures) with masking.

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    input_1 (InputLayer)         (None, 100, 18)           0
    _________________________________________________________________
    mask (Masking)               (None, 100, 18)           0
    _________________________________________________________________
    td_dense1 (TimeDistributed)  (None, 100, 32)           608
    _________________________________________________________________
    td_dense2 (TimeDistributed)  (None, 100, 32)           1056
    _________________________________________________________________
    track_gru (GRU)              (None, 100, 32)           6240
    _________________________________________________________________
    noseq_gru (GRU)              (None, 32)                6240
    _________________________________________________________________
    out_dense_1 (Dense)          (None, 32)                1056
    _________________________________________________________________
    out_dense_2 (Dense)          (None, 32)                1056
    _________________________________________________________________
    out_dense_3 (Dense)          (None, 32)                1056
    _________________________________________________________________
    outputTag (Dense)            (None, 1)                 33
    =================================================================
    Total params: 17,345
    Trainable params: 17,345
    Non-trainable params: 0
    _________________________________________________________________

    '''

    trackInput =Klayers.Input(trackShape)
    tracks =Klayers.Masking(mask_value = -999, name = 'mask')(trackInput)

    tracks =Klayers.TimeDistributed(Klayers.Dense(32, activation = 'relu'), name = 'td_dense1')(tracks)
    tracks =Klayers.TimeDistributed(Klayers.Dense(32, activation = 'relu'), name = 'td_dense2')(tracks)

    tracks =Klayers.GRU(32, activation = 'relu', return_sequences = True, name = 'track_gru')(tracks)
    tracks =Klayers.GRU(32, activation = 'relu', return_sequences = False, recurrent_dropout = 0.5, name = 'noseq_gru')(tracks)

    tracks =Klayers.Dense(27, activation = 'relu', name = 'out_dense_1')(tracks)
    tracks =Klayers.Dense(27, activation = 'relu', name = 'out_dense_2')(tracks)
    tracks =Klayers.Dense(27, activation = 'relu', name = 'out_dense_3')(tracks)
    tracks =Klayers.Dense(27, activation = 'relu', name = 'out_dense_4')(tracks)
    outputTag =Klayers.Dense(1, activation='sigmoid', name = 'outputTag')(tracks)

    return Model(inputs = trackInput, outputs = outputTag)

def tagNetworkVeryDeep(trackShape, units):

    '''

    Tag classifier taking input shape (nBatch, nTracks, nFeatures) with masking.

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    input_1 (InputLayer)         (None, 100, 18)           0
    _________________________________________________________________
    mask (Masking)               (None, 100, 18)           0
    _________________________________________________________________
    td_dense1 (TimeDistributed)  (None, 100, 32)           608
    _________________________________________________________________
    td_dense2 (TimeDistributed)  (None, 100, 32)           1056
    _________________________________________________________________
    track_gru (GRU)              (None, 100, 32)           6240
    _________________________________________________________________
    noseq_gru (GRU)              (None, 32)                6240
    _________________________________________________________________
    out_dense_1 (Dense)          (None, 32)                1056
    _________________________________________________________________
    out_dense_2 (Dense)          (None, 32)                1056
    _________________________________________________________________
    out_dense_3 (Dense)          (None, 32)                1056
    _________________________________________________________________
    outputTag (Dense)            (None, 1)                 33
    =================================================================
    Total params: 17,345
    Trainable params: 17,345
    Non-trainable params: 0
    _________________________________________________________________

    '''

    trackInput =Klayers.Input(trackShape)
    tracks =Klayers.Masking(mask_value = -999, name = 'mask')(trackInput)

    tracks =Klayers.TimeDistributed(Klayers.Dense(32, activation = 'relu'), name = 'td_dense1')(tracks)
    tracks =Klayers.TimeDistributed(Klayers.Dense(32, activation = 'relu'), name = 'td_dense2')(tracks)

    tracks =Klayers.GRU(32, activation = 'relu', return_sequences = True, name = 'track_gru')(tracks)
    tracks =Klayers.GRU(32, activation = 'relu', return_sequences = False, recurrent_dropout = 0.5, name = 'noseq_gru')(tracks)

    tracks =Klayers.Dense(24, activation = 'relu', name = 'out_dense_1')(tracks)
    tracks =Klayers.Dense(24, activation = 'relu', name = 'out_dense_2')(tracks)
    tracks =Klayers.Dense(24, activation = 'relu', name = 'out_dense_3')(tracks)
    tracks =Klayers.Dense(24, activation = 'relu', name = 'out_dense_4')(tracks)
    tracks =Klayers.Dense(24, activation = 'relu', name = 'out_dense_5')(tracks)
    outputTag =Klayers.Dense(1, activation='sigmoid', name = 'outputTag')(tracks)

    return Model(inputs = trackInput, outputs = outputTag)

def tagNetworkRegL2(trackShape, units):

    '''

    Tag classifier taking input shape (nBatch, nTracks, nFeatures) with masking.

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    input_1 (InputLayer)         (None, 100, 18)           0
    _________________________________________________________________
    mask (Masking)               (None, 100, 18)           0
    _________________________________________________________________
    td_dense1 (TimeDistributed)  (None, 100, 32)           608
    _________________________________________________________________
    td_dense2 (TimeDistributed)  (None, 100, 32)           1056
    _________________________________________________________________
    track_gru (GRU)              (None, 100, 32)           6240
    _________________________________________________________________
    noseq_gru (GRU)              (None, 32)                6240
    _________________________________________________________________
    out_dense_1 (Dense)          (None, 32)                1056
    _________________________________________________________________
    out_dense_2 (Dense)          (None, 32)                1056
    _________________________________________________________________
    out_dense_3 (Dense)          (None, 32)                1056
    _________________________________________________________________
    outputTag (Dense)            (None, 1)                 33
    =================================================================
    Total params: 17,345
    Trainable params: 17,345
    Non-trainable params: 0
    _________________________________________________________________

    '''

    trackInput =Klayers.Input(trackShape)
    tracks =Klayers.Masking(mask_value = -999, name = 'mask')(trackInput)

    tracks =Klayers.TimeDistributed(Klayers.Dense(32, activation = 'relu'), name = 'td_dense1')(tracks)
    tracks =Klayers.TimeDistributed(Klayers.Dense(32, activation = 'relu'), name = 'td_dense2')(tracks)

    tracks =Klayers.GRU(32, activation = 'relu', return_sequences = True, name = 'track_gru')(tracks)
    tracks =Klayers.GRU(32, activation = 'relu', return_sequences = False, recurrent_dropout = 0.5, name = 'noseq_gru')(tracks)

    tracks =Klayers.Dense(32, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.l2(1e-4),
    activity_regularizer=regularizers.l2(1e-5), activation = 'relu', name = 'out_dense_1')(tracks)
    tracks =Klayers.Dense(32, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.l2(1e-4),
    activity_regularizer=regularizers.l2(1e-5), activation = 'relu', name = 'out_dense_2')(tracks)
    tracks =Klayers.Dense(32, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.l2(1e-4),
    activity_regularizer=regularizers.l2(1e-5),activation = 'relu', name = 'out_dense_3')(tracks)

    outputTag =Klayers.Dense(1, activation='sigmoid', name = 'outputTag')(tracks)

    return Model(inputs = trackInput, outputs = outputTag)

def tagNetworkDropout(trackShape, units):

    '''

    Tag classifier taking input shape (nBatch, nTracks, nFeatures) with masking.

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    input_1 (InputLayer)         (None, 100, 18)           0
    _________________________________________________________________
    mask (Masking)               (None, 100, 18)           0
    _________________________________________________________________
    td_dense1 (TimeDistributed)  (None, 100, 32)           608
    _________________________________________________________________
    td_dense2 (TimeDistributed)  (None, 100, 32)           1056
    _________________________________________________________________
    track_gru (GRU)              (None, 100, 32)           6240
    _________________________________________________________________
    noseq_gru (GRU)              (None, 32)                6240
    _________________________________________________________________
    out_dense_1 (Dense)          (None, 32)                1056
    _________________________________________________________________
    out_dense_2 (Dense)          (None, 32)                1056
    _________________________________________________________________
    out_dense_3 (Dense)          (None, 32)                1056
    _________________________________________________________________
    outputTag (Dense)            (None, 1)                 33
    =================================================================
    Total params: 17,345
    Trainable params: 17,345
    Non-trainable params: 0
    _________________________________________________________________

    '''

    trackInput =Klayers.Input(trackShape)
    tracks =Klayers.Masking(mask_value = -999, name = 'mask')(trackInput)

    tracks =Klayers.TimeDistributed(Klayers.Dense(32, activation = 'relu'), name = 'td_dense1')(tracks)
    tracks =Klayers.TimeDistributed(Klayers.Dense(32, activation = 'relu'), name = 'td_dense2')(tracks)

    tracks =Klayers.GRU(32, activation = 'relu', return_sequences = True, name = 'track_gru')(tracks)
    tracks =Klayers.GRU(32, activation = 'relu', return_sequences = False, recurrent_dropout = 0.5, name = 'noseq_gru')(tracks)

    tracks =Klayers.Dropout(units = 32, rate = 0.2, activation = 'relu', name = 'out_dense_1')(tracks)
    tracks =Klayers.Dropoit(units = 32, rate = 0.2, activation = 'relu', name = 'out_dense_2')(tracks)
    tracks =Klayers.Dropout(units =32, rate = 0.2, activation = 'relu', name = 'out_dense_3')(tracks)

    outputTag =Klayers.Dropout(units = 1, rate = 0.2, activation='sigmoid', name = 'outputTag')(tracks)

    return Model(inputs = trackInput, outputs = outputTag)

def tagNetworkEmbed(trackShape, nB):

    '''

    Tag classifier taking input shape (nBatch, nTracks, nFeatures) with masking, and a vector of nB
    training B type indices (enum(Bu, Bd, Bs)) to be embedded.

    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to
    ==================================================================================================
    input_1 (InputLayer)            (None, 100, 18)      0
    __________________________________________________________________________________________________
    mask (Masking)                  (None, 100, 18)      0           input_1[0][0]
    __________________________________________________________________________________________________
    td_dense1 (TimeDistributed)     (None, 100, 32)      608         mask[0][0]
    __________________________________________________________________________________________________
    input_2 (InputLayer)            (None, 1)            0
    __________________________________________________________________________________________________
    td_dense2 (TimeDistributed)     (None, 100, 32)      1056        td_dense1[0][0]
    __________________________________________________________________________________________________
    embedding_1 (Embedding)         (None, 1, 8)         32          input_2[0][0]
    __________________________________________________________________________________________________
    track_gru (GRU)                 (None, 100, 32)      6240        td_dense2[0][0]
    __________________________________________________________________________________________________
    flatten_1 (Flatten)             (None, 8)            0           embedding_1[0][0]
    __________________________________________________________________________________________________
    noseq_gru (GRU)                 (None, 32)           6240        track_gru[0][0]
    __________________________________________________________________________________________________
    embed_dense (Dense)             (None, 8)            72          flatten_1[0][0]
    __________________________________________________________________________________________________
    concatenate_1 (Concatenate)     (None, 40)           0           noseq_gru[0][0]
                                                                     embed_dense[0][0]
    __________________________________________________________________________________________________
    out_dense_1 (Dense)             (None, 32)           1312        concatenate_1[0][0]
    __________________________________________________________________________________________________
    out_dense_2 (Dense)             (None, 32)           1056        out_dense_1[0][0]
    __________________________________________________________________________________________________
    out_dense_3 (Dense)             (None, 32)           1056        out_dense_2[0][0]
    __________________________________________________________________________________________________
    outputTag (Dense)               (None, 1)            33          out_dense_3[0][0]
    ==================================================================================================
    Total params: 17,705
    Trainable params: 17,705
    Non-trainable params: 0
    __________________________________________________________________________________________________

    '''

    trackInput =Klayers.Input(trackShape)
    tracks =Klayers.Masking(mask_value = -999, name = 'mask')(trackInput)

    tracks =Klayers.TimeDistributed(Klayers.Dense(32, activation = 'relu'), name = 'td_dense1')(tracks)
    tracks =Klayers.TimeDistributed(Klayers.Dense(32, activation = 'relu'), name = 'td_dense2')(tracks)

    tracks =Klayers.GRU(32, activation = 'relu', return_sequences = True, name = 'track_gru')(tracks)
    tracks =Klayers.GRU(32, activation = 'relu', return_sequences = False, recurrent_dropout = 0.5, name = 'noseq_gru')(tracks)

    bTypeInput =Klayers.Input((1,))
    bType =Klayers.Embedding(nB + 1, 8)(bTypeInput) # output -> (batch_size, 1, 8)
    bType =Klayers.Flatten()(bType)
    bType =Klayers.Dense(8, activation = 'relu', name = 'embed_dense')(bType)

    # Try also concatenating to the time axis?

    tracks =Klayers.Concatenate(-1)([tracks, bType])

    tracks =Klayers.Dense(32, activation = 'relu', name = 'out_dense_1')(tracks)
    tracks =Klayers.Dense(32, activation = 'relu', name = 'out_dense_2')(tracks)
    tracks =Klayers.Dense(32, activation = 'relu', name = 'out_dense_3')(tracks)

    outputTag =Klayers.Dense(1, activation='sigmoid', name = 'outputTag')(tracks)

    return Model(inputs = [trackInput, bTypeInput], outputs = outputTag)

def tagNetworkEmbedSmall(trackShape, nB):

    '''

    Tag classifier taking input shape (nBatch, nTracks, nFeatures) with masking, and a vector of nB
    training B type indices (enum(Bu, Bd, Bs)) to be embedded.

    Reduced number of layers and parameters, for testing.

    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to
    ==================================================================================================
    input_1 (InputLayer)            (None, 100, 18)      0
    __________________________________________________________________________________________________
    input_2 (InputLayer)            (None, 1)            0
    __________________________________________________________________________________________________
    mask (Masking)                  (None, 100, 18)      0           input_1[0][0]
    __________________________________________________________________________________________________
    embedding_1 (Embedding)         (None, 1, 4)         16          input_2[0][0]
    __________________________________________________________________________________________________
    td_dense1 (TimeDistributed)     (None, 100, 16)      304         mask[0][0]
    __________________________________________________________________________________________________
    flatten_1 (Flatten)             (None, 4)            0           embedding_1[0][0]
    __________________________________________________________________________________________________
    noseq_gru (GRU)                 (None, 16)           1584        td_dense1[0][0]
    __________________________________________________________________________________________________
    embed_dense (Dense)             (None, 4)            20          flatten_1[0][0]
    __________________________________________________________________________________________________
    concatenate_1 (Concatenate)     (None, 20)           0           noseq_gru[0][0]
                                                                     embed_dense[0][0]
    __________________________________________________________________________________________________
    out_dense_1 (Dense)             (None, 16)           336         concatenate_1[0][0]
    __________________________________________________________________________________________________
    outputTag (Dense)               (None, 1)            17          out_dense_1[0][0]
    ==================================================================================================
    Total params: 2,277
    Trainable params: 2,277
    Non-trainable params: 0
    __________________________________________________________________________________________________

    '''

    trackInput =Klayers.Input(trackShape)
    tracks =Klayers.Masking(mask_value = -999, name = 'mask')(trackInput)

    tracks =Klayers.TimeDistributed(Klayers.Dense(16, activation = 'relu'), name = 'td_dense1')(tracks)

    tracks =Klayers.GRU(16, activation = 'relu', return_sequences = False, recurrent_dropout = 0.5, name = 'noseq_gru')(tracks)

    bTypeInput =Klayers.Input((1,))
    bType =Klayers.Embedding(nB + 1, 4)(bTypeInput) # output -> (batch_size, 1, 4)
    bType =Klayers.Flatten()(bType)
    bType =Klayers.Dense(4, activation = 'relu', name = 'embed_dense')(bType)

    # Try also concatenating to the time axis?

    tracks =Klayers.Concatenate(-1)([tracks, bType])

    tracks =Klayers.Dense(16, activation = 'relu', name = 'out_dense_1')(tracks)

    outputTag =Klayers.Dense(1, activation='sigmoid', name = 'outputTag')(tracks)

    return Model(inputs = [trackInput, bTypeInput], outputs = outputTag)

def tagCatNetworkFlat(trackShape, trackCategories):

    '''

    Combined tag and track category network, with no post-processing of the track categories (cuts, etc).

    To enable the track category part of the network to be used to select input tracks, and be fast to evaluate,
    this operates on a flat list of tracks with no recurrent layers, and the output is transformed on the fly
    to be concatenated to the tag part of the network.

    This may improve performance of the track category component, as by being trained at the same time as the tag
    information, the information on how the track categories affect the tag can be back-propagated.

    The structure here enables the track category part to take a flat list of tracks, but the tag part can still
    make use of the event information with the recurrent layer, through backpropagation.

    The input to the 'tag' part of the network is from the mask layer (not from the 'category') part of the
    network, so that these can be separated.

    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to
    ==================================================================================================
    input_1 (InputLayer)            (None, 100, 18)      0
    __________________________________________________________________________________________________
    mask (Masking)                  (None, 100, 18)      0           input_1[0][0]
    __________________________________________________________________________________________________
    lambda_1 (Lambda)               (None, 18)           0           mask[0][0]
    __________________________________________________________________________________________________
    cat_dense1 (Dense)              (None, 32)           608         lambda_1[0][0]
    __________________________________________________________________________________________________
    cat_dense2 (Dense)              (None, 32)           1056        cat_dense1[0][0]
    __________________________________________________________________________________________________
    dropout_1 (Dropout)             (None, 32)           0           cat_dense2[0][0]
    __________________________________________________________________________________________________
    cat_dense3 (Dense)              (None, 32)           1056        dropout_1[0][0]
    __________________________________________________________________________________________________
    cat_dense4 (Dense)              (None, 32)           1056        cat_dense3[0][0]
    __________________________________________________________________________________________________
    outputCat (Dense)               (None, 4)            132         cat_dense4[0][0]
    __________________________________________________________________________________________________
    lambda_2 (Lambda)               (None, 100, 4)       0           outputCat[0][0]
    __________________________________________________________________________________________________
    concatenate_1 (Concatenate)     (None, 100, 22)      0           mask[0][0]
                                                                     lambda_2[0][0]
    __________________________________________________________________________________________________
    tag_td_dense1 (TimeDistributed) (None, 100, 32)      736         concatenate_1[0][0]
    __________________________________________________________________________________________________
    tag_td_dense2 (TimeDistributed) (None, 100, 32)      1056        tag_td_dense1[0][0]
    __________________________________________________________________________________________________
    track_gru (GRU)                 (None, 100, 32)      6240        tag_td_dense2[0][0]
    __________________________________________________________________________________________________
    noseq_gru (GRU)                 (None, 32)           6240        track_gru[0][0]
    __________________________________________________________________________________________________
    tag_dense_1 (Dense)             (None, 32)           1056        noseq_gru[0][0]
    __________________________________________________________________________________________________
    tag_dense_2 (Dense)             (None, 32)           1056        tag_dense_1[0][0]
    __________________________________________________________________________________________________
    tag_dense_3 (Dense)             (None, 32)           1056        tag_dense_2[0][0]
    __________________________________________________________________________________________________
    outputTag (Dense)               (None, 1)            33          tag_dense_3[0][0]
    ==================================================================================================
    Total params: 21,381
    Trainable params: 21,381
    Non-trainable params: 0
    __________________________________________________________________________________________________

    '''

    trackInput =Klayers.Input(trackShape)

    tracks =Klayers.Masking(mask_value = -999, name = 'mask')(trackInput)

    # Reshape (nBatch, nTracks, nFeatures) -> (nBatch * nTracks, nFeatures)

    # Although this seems to throw the error:
    # 'ValueError: An operation has `None` for gradient.
    # Please make sure that all of your ops have a gradient defined (i.e. are differentiable)'

    tracksCat =Klayers.Lambda( lambda x : K.reshape(tracks, (-1, trackShape[1])) )(tracks)

    # Proceed as normal for 'flat' input for track categories

    tracksCat =Klayers.Dense(32, activation = 'relu', name = 'cat_dense1')(tracksCat)
    tracksCat =Klayers.Dense(32, activation = 'relu', name = 'cat_dense2')(tracksCat)

    tracksCat =Klayers.Dropout(0.5)(tracksCat)

    tracksCat =Klayers.Dense(32, activation = 'relu', name = 'cat_dense3')(tracksCat)
    tracksCat =Klayers.Dense(32, activation = 'relu', name = 'cat_dense4')(tracksCat)

    tracksCat =Klayers.Dense(trackCategories, activation = 'softmax', name = 'catSoftmax')(tracksCat)

    outputCat =Klayers.Lambda( lambda x : K.reshape(tracks, (-1, trackShape[0], trackCategories)), name = 'outputCat' )(tracksCat)

    # Concatenate track category features to track features, now size nFeatures + nTrackCategories

    tracks =Klayers.Concatenate()([tracks, outputCat])

    # Now proceed as normal

    tracks =Klayers.TimeDistributed(Klayers.Dense(32, activation = 'relu'), name = 'tag_td_dense1')(tracks)
    tracks =Klayers.TimeDistributed(Klayers.Dense(32, activation = 'relu'), name = 'tag_td_dense2')(tracks)

    tracks =Klayers.GRU(32, activation = 'relu', return_sequences = True, name = 'track_gru')(tracks)
    tracks =Klayers.GRU(32, activation = 'relu', return_sequences = False, recurrent_dropout = 0.5, name = 'noseq_gru')(tracks)

    tracks =Klayers.Dense(32, activation = 'relu', name = 'tag_dense_1')(tracks)
    tracks =Klayers.Dense(32, activation = 'relu', name = 'tag_dense_2')(tracks)
    tracks =Klayers.Dense(32, activation = 'relu', name = 'tag_dense_3')(tracks)

    outputTag =Klayers.Dense(1, activation='sigmoid', name = 'outputTag')(tracks)

    return Model(inputs = trackInput, outputs = [outputCat, outputTag])

def tagCatNetwork(trackShape, trackCategories):

    '''

    Combined tag and track category network.

    The track category output uses the same first GRU layer as the tag network, and the track category
    probabilities are concatenated after this point.

    This requires track categories to have the track dimension (i.e., shape = (nBatch, nTracks, nCategories)),
    rather than be flat, with the same shape for the category-wise weights.

    The input to the 'tag' part of the network is from the mask layer (not from the 'category') part of the
    network, so that these can be separated.

    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to
    ==================================================================================================
    input_1 (InputLayer)            (None, 100, 18)      0
    __________________________________________________________________________________________________
    mask (Masking)                  (None, 100, 18)      0           input_1[0][0]
    __________________________________________________________________________________________________
    td_dense1 (TimeDistributed)     (None, 100, 32)      608         mask[0][0]
    __________________________________________________________________________________________________
    td_dense2 (TimeDistributed)     (None, 100, 32)      1056        td_dense1[0][0]
    __________________________________________________________________________________________________
    track_gru (GRU)                 (None, 100, 32)      6240        td_dense2[0][0]
    __________________________________________________________________________________________________
    outputCat (Dense)               (None, 100, 4)       132         track_gru[0][0]
    __________________________________________________________________________________________________
    concatenate_1 (Concatenate)     (None, 100, 36)      0           track_gru[0][0]
                                                                     outputCat[0][0]
    __________________________________________________________________________________________________
    noseq_gru (GRU)                 (None, 32)           6624        concatenate_1[0][0]
    __________________________________________________________________________________________________
    out_dense_1 (Dense)             (None, 32)           1056        noseq_gru[0][0]
    __________________________________________________________________________________________________
    out_dense_2 (Dense)             (None, 32)           1056        out_dense_1[0][0]
    __________________________________________________________________________________________________
    out_dense_3 (Dense)             (None, 32)           1056        out_dense_2[0][0]
    __________________________________________________________________________________________________
    outputTag (Dense)               (None, 1)            33          out_dense_3[0][0]
    ==================================================================================================
    Total params: 17,861
    Trainable params: 17,861
    Non-trainable params: 0
    __________________________________________________________________________________________________

    '''

    trackInput =Klayers.Input(trackShape)

    tracksMasked =Klayers.Masking(mask_value = -999, name = 'mask')(trackInput)

    tracks =Klayers.TimeDistributed(Klayers.Dense(32, activation = 'relu'), name = 'td_dense1')(tracksMasked)
    tracks =Klayers.TimeDistributed(Klayers.Dense(32, activation = 'relu'), name = 'td_dense2')(tracks)

    tracks =Klayers.GRU(32, activation = 'relu', return_sequences = True, name = 'track_gru')(tracks)

    outputCat =Klayers.Dense(trackCategories, activation = 'softmax', name = 'outputCat')(tracks)

    # Concatenate with output from mask rather than 'tracks', so these networks can be separated

    tracks =Klayers.Concatenate()([tracksMasked, outputCat])

    tracks =Klayers.GRU(32, activation = 'relu', return_sequences = False, recurrent_dropout = 0.5, name = 'noseq_gru')(tracks)

    tracks =Klayers.Dense(32, activation = 'relu', name = 'out_dense_1')(tracks)
    tracks =Klayers.Dense(32, activation = 'relu', name = 'out_dense_2')(tracks)
    tracks =Klayers.Dense(32, activation = 'relu', name = 'out_dense_3')(tracks)

    outputTag =Klayers.Dense(1, activation='sigmoid', name = 'outputTag')(tracks)
    # outputTag =Klayers.Reshape((1, 1))(outputTag)

    return Model(inputs = trackInput, outputs = [outputCat, outputTag])

if __name__ == '__main__':

    # model = catNetwork((100, 18,), 4)
    # model = tagNetworkEmbed((100, 18), 3)
    model = tagNetworkSmall((100, 18))
    model.summary()
