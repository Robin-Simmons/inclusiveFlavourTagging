"""
Train a simple RNN model for Inclusive Flavour Tagging using track information.
- Using DataGenerator class to stream data to the GPU, to avoid storing loads of data in RAM
- No transformations on data are done in this part now - these are all done per-batch in DataGenerator
"""

__author__ = "Daniel O'Hanlon <daniel.ohanlon@cern.ch>"

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from sklearn.metrics import roc_auc_score

import argparse
import time
from ift_tf_2.training import modelDefinition

from ift_tf_2.training.dataGenerator import createSplitGenerators

from ift_tf_2.utils.utils import decision_and_mistag, saveModel, exportForCalibration
from ift_tf_2.utils.plotUtils import makeTrainingPlots
import os
TRACK_SHAPE = (100, 18)

generatorOptions = {

# Classify:
# 'tag' -> one per event
# 'category' ->  one per track
# 'category_flat' -> one per track with flattened inputs and outputs

'trainingType' : 'tag',

# Location of the features (per track) in .h5 file

'featureName' : 'featureArray',

# Location of the tag (per event) in .h5 file

'tagName' : 'tagArray',

'nFeatures' : 18,

# Need to configure each of these in all three to avoid overlaps
# With a different file/dataset, can set these to 1.0 separately if no
# training/testing will be done

'trainFrac' : 0.8, #change back to 0.8
'validationFrac' : 0.1,
'testFrac' : 0.1,
}

def train(args):

    # Now controls host RAM usage too -> optimise for GPU/host RAM and execution speed
    generatorOptions['batchSize'] = args.batchSize

    if args.nEvents : generatorOptions['dataSize'] = args.nEvents

    # Whether to use multiprocessing for parallel data loading
    generatorOptions['useMultiprocessing'] = args.useMultiprocessing

    network = getattr(modelDefinition, args.network)

    model = network(TRACK_SHAPE, args.nHidden)
    if args.verbose : model.summary()

    adam = Adam(lr = args.learningRate, amsgrad = True)
    earlyStopping = EarlyStopping(patience = args.patience)
    model.summary()
    model.compile(optimizer = adam, loss = 'binary_crossentropy', metrics=['accuracy'])

    genTrain, genValidation, genTest = createSplitGenerators(args.inputFiles,
                                                             generatorOptions,
                                                             shuffle = args.shuffle or args.shuffleChunks,
                                                             shuffleChunks = args.shuffleChunks)
    
    history = model.fit_generator(generator = genTrain,
                        validation_data = genValidation,
                        use_multiprocessing = args.useMultiprocessing,
                        workers = args.nWorkers,
                        callbacks = [earlyStopping],
                        epochs = args.epochs, verbose = True)
    
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
        
    #makeTrainingPlots(model, plotdir = args.outputDir, modelName = args.modelName)
    #makeTrainingPlots(model, plotdir = args.outputDir, modelName = args.modelName)
    saveModel(model, args.outputDir + args.modelName)
    exportForCalibration(y_test, y_out_test, args.outputDir)
    return model, history

def prune(model, args, finalSparsity):
    import tensorflow_model_optimization as tfmot 
    # Now controls host RAM usage too -> optimise for GPU/host RAM and execution speed
    generatorOptions['batchSize'] = args.batchSize

    if args.nEvents : generatorOptions['dataSize'] = args.nEvents

    # Whether to use multiprocessing for parallel data loading
    generatorOptions['useMultiprocessing'] = args.useMultiprocessing

    prune_low_mag = tfmot.sparsity.keras.prune_low_magnitude

    pruning_params = {"pruning_schedule" :
                      tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=args.startSparsity,
                                                                   final_sparsity=finalSparsity,
                                                                   begin_step=0,
                                                                   end_step=args.epochs*args.batchSize)
    }

    prune_model = prune_low_mag(model, **pruning_params)    
    adam = Adam(lr = args.learningRate/10, amsgrad = True)
    earlyStopping = EarlyStopping(patience = args.patience)

    prune_model.compile(optimizer = adam, loss = 'binary_crossentropy', metrics=['accuracy'])

    genTrain, genValidation, genTest = createSplitGenerators(args.inputFiles,
                                                             generatorOptions,
                                                             shuffle = args.shuffle or args.shuffleChunks,
                                                             shuffleChunks = args.shuffleChunks)
    prune_model.summary()
    log_dir = "./prune"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),
    tfmot.sparsity.keras.PruningSummaries(log_dir=log_dir)
    ]
    prune_history = prune_model.fit_generator(generator = genTrain,
                        validation_data = genValidation,
                        use_multiprocessing = args.useMultiprocessing,
                        workers = args.nWorkers,
                        callbacks = callbacks,
                        epochs = args.pruneEpochs, verbose = False)
    
    # Get the tags for the full training sample, so that these can be used to calculate the ROC
    y_train = genTrain.getTags()
    y_test = genTest.getTags()

    # Can use the generators for prediction too, but need to ensure that there is no shuffling wrt the above
    y_out_train = model.predict_generator(genTrain)
    y_out_test = model.predict_generator(genTest)

    rocAUC_train = roc_auc_score(y_train, y_out_train)
    rocAUC_test = roc_auc_score(y_test, y_out_test)

    print('Prune {}  ROC Train: {}'.format(finalSparsity, rocAUC_train))
    print('Prune {} ROC Test: {}'.format(finalSparsity, rocAUC_test))
    stripped_model = tfmot.sparsity.keras.strip_pruning(prune_model)    
    #makeTrainingPlots(model, plotdir = args.outputDir, modelName = args.modelName)
    #makeTrainingPlots(model, plotdir = args.outputDir, modelName = args.modelName)
    saveModel(model, args.outputDir + args.modelName)
    exportForCalibration(y_test, y_out_test, args.outputDir)
    return stripped_model, prune_history

def time_inference(model, name):
    model.summary()
    with tf.device("/cpu:0"):
        times = []
        for i in range(args.evalRepeats):
            event = np.random.random((1000,100,18))
            start = time.time()
            model.predict(event)
            times.append(time.time()-start)
    print("Model {} took {} +/- {} s to evaluate 1 event".format(name, np.mean(times), np.std(times)))
if __name__ == '__main__':

    argParser = argparse.ArgumentParser()

    argParser.add_argument("-f", "--inputFiles", type = str, dest = "inputFiles", nargs = '+', default = "DTT_MC2015_Reco15aStrip24_DIMUON_Bd2JpsiKstar.h5", help = 'Input file names.')
    argParser.add_argument("-e", "--epochs", type = int, dest = "epochs", default = 1, help = 'Number of epochs to train for.')
    argParser.add_argument("-n", "--name", type = str, dest = "modelName", default = "tag", help = 'Model name.')
    argParser.add_argument("-v", "--verbose", default = False, action = "store_true", dest = "verbose", help = 'Verbose mode on.')
    argParser.add_argument("-b", "--batchSize", type = int, dest = "batchSize", default = 2 ** 14, help = 'Training batch size.')
    argParser.add_argument("-l", "--learningRate", type = float, dest = "learningRate", default = 1E-3, help = 'Adam learning rate.')

    argParser.add_argument("--network", type = str, dest = "network", default = "tagNetwork", help = 'Network model name.')

    argParser.add_argument("--nEvents", type = int, dest = "nEvents", default = None, help = 'Number of total events to evaluate on.')
    argParser.add_argument("--shuffle", default = False, action = "store_true", dest = "shuffle", help = 'Shuffle traning data.')
    argParser.add_argument("--shuffleChunks", default = False, action = "store_true", dest = "shuffleChunks", help = 'Shuffle training data according to chunks.')
    argParser.add_argument("--nCPU", type = int, dest = "nCPU", default = 1, help = 'Number of threads to use.')
    argParser.add_argument("--nWorkers", type = int, dest = "nWorkers", default = 1, help = 'Number of data loaders.')
    argParser.add_argument("--useMultiprocessing", default = False, action = "store_true", dest = "useMultiprocessing", help = 'Use multiprocessing with nWorkers workers.')
    argParser.add_argument("--patience", type = int, dest = "patience", default = 100, help = 'Early stopping patience (epochs).')
    argParser.add_argument("--outputDir", type = str, dest = "outputDir", default = "./", help = 'Directory to store model and plots')
    argParser.add_argument("--nHidden", type = int, dest = "nHidden", default = 16, help = "Parameter to vary")
    argParser.add_argument("--evalRepeats", type = int, dest = "evalRepeats", default = 5000, help = "Number of times to repeat measurment of evaluation time")
    argParser.add_argument("--sample", type = str, dest = "sample", default = "validation", help = 'Chosen sample: validation (default), test or training')
    argParser.add_argument("--appendTo", type = str, dest = "appendTo", default = None, help = 'HDF5 file with a single Pandas DataFrame to append the decision and mistag to.')
    argParser.add_argument("--evaluation", default = False, action = "store_true", dest = "evaluation", help = 'Run in evaluation mode over the full input dataset.')
    argParser.add_argument("--pruneEpochs", type = int, dest = "pruneEpochs", default = 25)
    argParser.add_argument("--startSparsity", type = float, dest = "startSparsity", default = 0.5)
    argParser.add_argument("--finalSparsity", type = float, dest = "finalSparsity", default = 0.8)
    args = argParser.parse_args()

    model, history = train(args)
    import matplotlib.pyplot as plt
    import numpy as np
    time_inference(model, "un pruned")
    for i in range(12,20):
        print("training with final sparsity {}".format(i/20))
        prune_model, prune_history = prune(model, args, i/20)
        time_inference(prune_model, "pruned")
    #acc = history.history["accuracy"]
    #val_acc = history.history["val_accuracy"]
    #prune_acc = prune_history.history["accuracy"]
    #prune_val_acc = prune_history.history["val_accuracy"]
    #acc_total = np.concatenate([acc, prune_acc])
    #val_acc_total = np.concatenate([val_acc, prune_val_acc])
    #fig = plt.figure(figsize = (12,9))
    #ax = fig.add_subplot(111)
    #plotdir = args.outputDir
    #plt.plot(acc_total, lw = 1.0)
    #plt.plot(val_acc_total, lw = 1.0)
    #plt.axvline(x = len(acc), lw = 1.0)
    #plt.xlabel("Epoch")
    #plt.ylabel("Accuracy")
    #plt.savefig(plotdir + 'acc-' + args.modelName + '.pdf')
    #plt.clf()
    
        
    print(args) 
