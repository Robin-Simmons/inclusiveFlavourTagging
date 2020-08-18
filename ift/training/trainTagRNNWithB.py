"""
Train a simple RNN model for Inclusive Flavour Tagging using track information.
- Using DataGenerator class to stream data to the GPU, to avoid storing loads of data in RAM
- No transformations on data are done in this part now - these are all done per-batch in DataGenerator
"""

__author__ = "Daniel O'Hanlon <daniel.ohanlon@cern.ch>"

from sklearn.metrics import roc_auc_score, roc_curve

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use(['fivethirtyeight', 'seaborn-whitegrid', 'seaborn-ticks'])
import matplotlib.ticker as plticker

from keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping
from keras.optimizers import Adam

from ift.utils.utils import decision_and_mistag, saveModel, exportForCalibration
from ift.utils.plotUtils import makeTrainingPlots

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from modelDefinition import tagNetworkEmbed

import shelve

from dataGenerator import createSplitGenerators, DataGenerator

#Training configuration
maxtracks      = 100
epochs         = 100
batch_size     = 2 ** 12

TRACK_SHAPE = (100, 18) # nTracks, nFeatures

nB = 2 # B+ and B0

model = tagNetworkEmbed(TRACK_SHAPE, nB)
model.summary()

adam = Adam(lr = 0.001, amsgrad = True)
earlyStopping = EarlyStopping(patience = 100)

model.compile(optimizer = adam, loss = 'binary_crossentropy', metrics=['accuracy'])

generatorOptions = {
'trainingType' : 'tag_plus_extra',

'featureName' : 'featureArray',

'tagName' : 'tagArray',

'extrasFeatureName' : 'bTypeArray',

'catName' : 'catArray',

'nFeatures' : 19,

'trainFrac' : 0.8,
'validationFrac' : 0.1,
'testFrac' : 0.1,

'batchSize' : batch_size,

}

# inputFiles = ['/home/dan/ftData/DTT_MC2015_Reco15aStrip24_Up_DIMUON_Bd2JpsiKstar_WithBType.h5',
#               '/home/dan/ftData/DTT_MC2016_Reco16Strip26_Up_DIMUON_Bu2JpsiK_WithBType.h5']

inputFiles = ['/home/dan/ftData/DTT_MC2015_Reco15aStrip24_Up_DIMUON_Bd2JpsiKstar_WithBType_100k.h5',
              '/home/dan/ftData/DTT_MC2016_Reco16Strip26_Up_DIMUON_Bu2JpsiK_WithBType_100k.h5']

# inputFiles = '/home/dan/ftData/DTT_MC2015_Reco15aStrip24_Up_DIMUON_Bd2JpsiKstar_WithBType_100k.h5'

genTrain, genValidation, genTest = createSplitGenerators(inputFiles,
                                                         generatorOptions,
                                                         shuffle = False,
                                                         shuffleChunks = False,
                                                         frac = 1.0)

model.fit_generator(generator = genTrain,
                    validation_data = genValidation,
                    callbacks = [earlyStopping],
                    epochs = 100, verbose = 1)

y_train = genTrain.getTags()
y_test = genTest.getTags()

y_out_train = model.predict_generator(genTrain)
y_out_test = model.predict_generator(genTest)

rocAUC_train = roc_auc_score(y_train, y_out_train)
rocAUC_test = roc_auc_score(y_test, y_out_test)

print('ROC Train:', rocAUC_train)
print('ROC Test:', rocAUC_test)

validFileBu = '/home/dan/ftData/DTT_MC2016_Reco16Strip26_Up_DIMUON_Bu2JpsiK_WithBType.h5'
validFileBd = '/home/dan/ftData/DTT_MC2015_Reco15aStrip24_Up_DIMUON_Bd2JpsiKstar_WithBType.h5'

genBu = DataGenerator(validFileBu, **generatorOptions, dataset = 'evaluation')
genBd = DataGenerator(validFileBd, **generatorOptions, dataset = 'evaluation')

y_Bu = genBu.getTags()
y_Bd = genBd.getTags()

y_out_Bu = model.predict_generator(genBu)
y_out_Bd = model.predict_generator(genBd)

rocAUC_Bu = roc_auc_score(y_Bu, y_out_Bu)
rocAUC_Bd = roc_auc_score(y_Bd, y_out_Bd)

print('ROC Bu:', rocAUC_Bu)
print('ROC Bd', rocAUC_Bd)
