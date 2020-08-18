"""
Train a model for to predict track (OS, SS frag., OS frag., background) categories for inclusive tagging
- Using DataGenerator class to stream data to the GPU, to avoid storing loads of data in RAM
- No transformations on data are done in this part now - these are all done per-batch in DataGenerator
"""

__author__ = "Daniel O'Hanlon <daniel.ohanlon@cern.ch>"

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use(['fivethirtyeight', 'seaborn-whitegrid', 'seaborn-ticks'])

from matplotlib import rcParams
rcParams['axes.facecolor'] = 'FFFFFF'
rcParams['savefig.facecolor'] = 'FFFFFF'
rcParams['figure.facecolor'] = 'FFFFFF'
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'

from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, confusion_matrix

from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

import numpy as np

import argparse

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from ift.training.modelDefinition import catNetwork, catNetworkFlat

from ift.training.dataGenerator import createSplitGenerators
from ift.utils.utils import flatten, evaluateOvRScore, evaluateOvRPredictions
from ift.utils.utils import getOvRPredictions, exportForCalibration, saveModel
from ift.utils.plotUtils import plot_confusion_matrix, makeTrainingPlots

nTrackCategories = 4
TRACK_SHAPE = (100, 18)

generatorOptions = {

'nFeatures' : 18,

'nClasses' : nTrackCategories,

'trainFrac' : 0.8,
'validationFrac' : 0.1,
'testFrac' : 0.1,
}

def train(args):

    # Now controls host RAM usage too -> optimise for GPU/host RAM and execution speed
    generatorOptions['batchSize'] = args.batchSize

    generatorOptions['dataSize'] = args.nEvents
    generatorOptions['useWeights'] = args.useWeights

    generatorOptions['trainingType'] = 'category' if not args.flat else 'category_flat'
    generatorOptions['featureName'] = 'featureArray' if not args.flat else 'featureArrayFlat'
    generatorOptions['catName'] = 'catArray' if not args.flat else 'catArrayFlat'

    if not args.flat:
        model = catNetwork(TRACK_SHAPE, nTrackCategories)
    else:
        model = catNetworkFlat(TRACK_SHAPE[1:], nTrackCategories)

    if args.verbose : model.summary()

    adam = Adam(lr = args.learningRate, amsgrad = True)
    earlyStopping = EarlyStopping(patience = args.patience)

    if not args.flat:
        model.compile(optimizer = adam, loss = 'categorical_crossentropy',
                      metrics=['categorical_accuracy'], sample_weight_mode = "temporal")
    else:
        model.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics=['categorical_accuracy'])

    genTrain, genValidation, genTest = createSplitGenerators(args.inputFiles,
                                                             generatorOptions,
                                                             shuffle = args.shuffle or args.shuffleChunks,
                                                             shuffleChunks = args.shuffleChunks)

    model.fit_generator(generator = genTrain,
                        validation_data = genValidation,
                        callbacks = [earlyStopping],
                        epochs = args.epochs, verbose = args.verbose)

    y_train = genTrain.getCats()
    y_test = genTest.getCats()

    # Can use the generators for prediction too, but need to ensure that there is no shuffling wrt the above
    y_out_train = model.predict_generator(genTrain)
    y_out_test = model.predict_generator(genTest)

    y_train = flatten(y_train)
    y_test = flatten(y_test)

    y_out_train = flatten(y_out_train)
    y_out_test = flatten(y_out_test)

    y_train_sparse = np.argmax(y_train, axis = 1)
    y_out_train_sparse = np.argmax(y_out_train, axis = 1)

    evaluateOvRPredictions(y_train, y_out_train, 'Train')
    evaluateOvRPredictions(y_test, y_out_test, 'Test')

    plot_confusion_matrix(y_train_sparse, y_out_train_sparse, np.unique(y_train_sparse), normalize = True)
    plt.savefig('confusion_matrix_' + args.modelName + '.pdf')
    plt.clf()

    makeTrainingPlots(model, accName = 'categorical_accuracy', modelName = args.modelName)
    saveModel(model, args.modelName)

if __name__ == '__main__':

    argParser = argparse.ArgumentParser()

    argParser.add_argument("-f", "--inputFiles", type = str, dest = "inputFiles", nargs = '+', default = "DTT_MC2015_Reco15aStrip24_DIMUON_Bd2JpsiKstar.h5", help = 'Input file names.')
    argParser.add_argument("-e", "--epochs", type = int, dest = "epochs", default = 1, help = 'Number of epochs to train for.')
    argParser.add_argument("-n", "--name", type = str, dest = "modelName", default = "cat", help = 'Model name.')
    argParser.add_argument("-v", "--verbose", default = False, action = "store_true", dest = "verbose", help = 'Verbose mode on.')
    argParser.add_argument("-b", "--batchSize", type = int, dest = "batchSize", default = 2 ** 14, help = 'Training batch size.')
    argParser.add_argument("-l", "--learningRate", type = int, dest = "learningRate", default = 1E-3, help = 'Adam learning rate.')

    argParser.add_argument("--nEvents", type = int, dest = "nEvents", default = None, help = 'Number of total events to evaluate on.')
    argParser.add_argument("--shuffle", default = False, action = "store_true", dest = "shuffle", help = 'Shuffle traning data.')
    argParser.add_argument("--useWeights", default = False, action = "store_true", dest = "useWeights", help = 'Use track category weights.')
    argParser.add_argument("--shuffleChunks", default = False, action = "store_true", dest = "shuffleChunks", help = 'Shuffle training data according to chunks.')
    argParser.add_argument("--nCPU", type = int, dest = "nCPU", default = 1, help = 'Number of threads to use.')
    argParser.add_argument("--patience", type = int, dest = "patience", default = 100, help = 'Early stopping patience (epochs).')
    argParser.add_argument("--flat", dest = "flat", default = False, action = "store_true", help = 'Train on pre-flattened arrays.')

    args = argParser.parse_args()

    from keras import backend as K
    K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=args.nCPU, inter_op_parallelism_threads=args.nCPU)))

    train(args)
