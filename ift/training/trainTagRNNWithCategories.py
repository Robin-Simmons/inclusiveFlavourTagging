"""
Train a simple RNN model for Inclusive Flavour Tagging using track information.
- Same as trainTagRNN.py, but with additional track category estimate inputs, trained using
  trainCategory(Flat)RNN.py and file populated with utils/fillScalingParamsForLWTNN.py
"""

__author__ = "Daniel O'Hanlon <daniel.ohanlon@cern.ch>"

from sklearn.metrics import roc_auc_score, roc_curve

from keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping
from keras.optimizers import Adam

from ift.utils.utils import decision_and_mistag, saveModel, exportForCalibration
from ift.utils.plotUtils import makeTrainingPlots

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from modelDefinition import tagNetwork

import shelve

from dataGenerator import createSplitGenerators

import argparse

nFeatures = 22 # nFeatures + nTrackCategories

TRACK_SHAPE = (100, nFeatures)

generatorOptions = {

# 'tag_plus_category' -> one tag per event, with input categories
'trainingType' : 'tag_plus_category',

# Location of the features (per track) in .h5 file
'featureName' : 'featureArray',

# Location of the tag (per event) in .h5 file
'tagName' : 'tagArray',

# Use extra (track category features from extrasFileName)
'useExtraFeatures' : True,

# Dataset name for category prediction features
'extrasFeatureName' : 'categoryPredictions',

# 18 features from inputFile, 4 from categoryFile
'nFeatures' : 22,

# Need to configure each of these in all three to avoid overlaps
# With a different file/dataset, can set these to 1.0 separately if no
# training/testing will be done

'trainFrac' : 0.8,
'validationFrac' : 0.1,
'testFrac' : 0.1,

}

def train(args):

    # Now controls host RAM usage too -> optimise for GPU/host RAM and execution speed
    generatorOptions['batchSize'] = args.batchSize

    generatorOptions['extrasFileName'] = args.categoryFile

    if args.nEvents : generatorOptions['dataSize'] = args.nEvents

    model = tagNetwork(TRACK_SHAPE)
    if args.verbose : model.summary()

    adam = Adam(lr = args.learningRate, amsgrad = True)
    earlyStopping = EarlyStopping(patience = args.patience)

    model.compile(optimizer = adam, loss = 'binary_crossentropy', metrics=['accuracy'])

    genTrain, genValidation, genTest = createSplitGenerators(args.inputFiles,
                                                             generatorOptions,
                                                             shuffle = args.shuffle or args.shuffleChunks,
                                                             shuffleChunks = args.shuffleChunks)

    model.fit_generator(generator = genTrain,
                        validation_data = genValidation,
                        callbacks = [earlyStopping],
                        epochs = args.epochs, verbose = args.verbose)

    # Get the tags for the full training sample, so that these can be used to calculate the ROC
    y_train = genTrain.getTags()
    y_test = genTest.getTags()

    # Can use the generators for prediction too, but need to ensure that there is no shuffling wrt the above
    y_out_train = model.predict_generator(genTrain)
    y_out_test = model.predict_generator(genTest)

    rocAUC_train = roc_auc_score(y_train, y_out_train)
    rocAUC_test = roc_auc_score(y_test, y_out_test)

    print(('ROC Train:', rocAUC_train))
    print(('ROC Test:', rocAUC_test))

    makeTrainingPlots(model)
    saveModel(model, args.modelName)
    exportForCalibration(y_test, y_out_test)

if __name__ == '__main__':

    argParser = argparse.ArgumentParser()

    argParser.add_argument("-f", "--inputFiles", type = str, dest = "inputFiles", nargs = '+', default = "DTT_MC2015_Reco15aStrip24_DIMUON_Bd2JpsiKstar.h5", help = 'Input file names.')
    argParser.add_argument("-c", "--categoryFile", type = str, dest = "categoryFile", default = "DTT_MC2015_Reco15aStrip24_DIMUON_Bd2JpsiKstar_Categories.h5", help = 'Input category file name.')
    argParser.add_argument("-e", "--epochs", type = int, dest = "epochs", default = 1, help = 'Number of epochs to train for.')
    argParser.add_argument("-n", "--name", type = str, dest = "modelName", default = "tag", help = 'Model name.')
    argParser.add_argument("-v", "--verbose", default = False, action = "store_true", dest = "verbose", help = 'Verbose mode on.')
    argParser.add_argument("-b", "--batchSize", type = int, dest = "batchSize", default = 2 ** 14, help = 'Training batch size.')
    argParser.add_argument("-l", "--learningRate", type = int, dest = "learningRate", default = 1E-3, help = 'Adam learning rate.')

    argParser.add_argument("--nEvents", type = int, dest = "nEvents", default = None, help = 'Number of total events to evaluate on.')
    argParser.add_argument("--shuffle", default = False, action = "store_true", dest = "shuffle", help = 'Shuffle traning data.')
    argParser.add_argument("--shuffleChunks", default = False, action = "store_true", dest = "shuffleChunks", help = 'Shuffle training data according to chunks.')
    argParser.add_argument("--nCPU", type = int, dest = "nCPU", default = 1, help = 'Number of threads to use.')
    argParser.add_argument("--patience", type = int, dest = "patience", default = 100, help = 'Early stopping patience (epochs).')

    args = argParser.parse_args()

    from keras import backend as K
    K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=args.nCPU, inter_op_parallelism_threads=args.nCPU)))

    train(args)
