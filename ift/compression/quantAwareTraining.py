"""
Run QAT on a convolutional model. There is also a function for fine tuning an existing model
"""

import numpy as np
import time
import tensorflow.keras as keras
import tensorflow_model_optimization as tfmot
import ift.training.modelDefinition
import tensorflow as tf
from ift.training.dataGenerator import createSplitGenerators, DataGenerator
from ift.conv import convDefinition
from ift.utils.utils import decision_and_mistag, saveModel, exportForCalibration
from ift.utils.plotUtils import makeTrainingPlots, makeTrainingPlotsTF2
import argparse
from sklearn.metrics import roc_auc_score
import ift.conv.convDefinition
import matplotlib.pyplot as plt
LastValueQuantizer = tfmot.quantization.keras.quantizers.LastValueQuantizer
MovingAverageQuantizer = tfmot.quantization.keras.quantizers.MovingAverageQuantizer

class DefaultDenseQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    # Configure how to quantize weights.
    def get_weights_and_quantizers(self, layer):
      return [(layer.kernel, LastValueQuantizer(num_bits=8, symmetric=True, narrow_range=False, per_axis=False))]

    # Configure how to quantize activations.
    def get_activations_and_quantizers(self, layer):
      return [(layer.activation, MovingAverageQuantizer(num_bits=8, symmetric=False, narrow_range=False, per_axis=False))]

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

quantize_annotate_layer = tfmot.quantization.keras.quantize_annotate_layer
quantize_annotate_model = tfmot.quantization.keras.quantize_annotate_model
quantize_scope = tfmot.quantization.keras.quantize_scope

trackShape = (100, 18)
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

'trainFrac' : 0.02, #change back to 0.8
'validationFrac' : 0.005,
'testFrac' : 0.005,
"forCNN" : True
}


def quant_aware_train(args):
    network = getattr(convDefinition, args.network)
    # Trains a randomly initilised model with Quantisation Aware Training
    model = network(trackShape, args)
    # `quantize_apply` requires mentioning `DefaultDenseQuantizeConfig` with `quantize_scope`

    
    with quantize_scope(
      {'DefaultDenseQuantizeConfig': DefaultDenseQuantizeConfig,
       "CustomDenseQuantizeConfig" : CustomDenseQuantizeConfig,
       }):
      # Use `quantize_apply` to actually make the model quantization aware.
      quant_aware_model = tfmot.quantization.keras.quantize_apply(model)
    
    if args.verbose == True: quant_aware_model.summary()
    # Compile and train model with early stopping 
    adam = keras.optimizers.Adam(lr = args.learningRate, amsgrad = True)
    earlyStopping = keras.callbacks.EarlyStopping(patience = args.patience)

    
    generatorOptions['batchSize'] = args.batchSize
    generatorOptions['useMultiprocessing'] = args.useMultiprocessing
    # Quantisation doesn't support sigmoid activations (yet) and so set from_logits = True
    quant_aware_model.compile(optimizer = "adam", loss = keras.losses.BinaryCrossentropy(from_logits=True), metrics = ["accuracy"])
    genTrain, genValidation, genTest = createSplitGenerators(args.inputFiles,
                                                                 generatorOptions,
                                                                 shuffle = args.shuffle or args.shuffleChunks,
                                                                 shuffleChunks = args.shuffleChunks)
    
    quant_aware_model.fit_generator(generator = genTrain,
                            validation_data = genValidation,
                            use_multiprocessing = args.useMultiprocessing,
                            workers = args.nWorkers,
                            epochs = args.epochs, verbose = args.verbose)

    y_train = genTrain.getTags()
    y_test = genTest.getTags()

    # Can use the generators for prediction too, but need to ensure that there is no shuffling wrt the above
    # Sigmoids needed to ensure outputs are in [0,1]
    y_out_train = keras.activations.sigmoid(quant_aware_model.predict_generator(genTrain))
    y_out_test = keras.activations.sigmoid(quant_aware_model.predict_generator(genTest))

    rocAUC_train = roc_auc_score(y_train, y_out_train)
    rocAUC_test = roc_auc_score(y_test, y_out_test)
    
    print(('ROC Train:', rocAUC_train))
    print(('ROC Test:', rocAUC_test))
    return quant_aware_model
    
def quant_aware_tune(args):
    print("Tuning existing model")
    network = getattr(convDefinition, args.network)
    # Fine tunes a pre-trained model with Quantisation Aware Training
    # Considered the "correct" method
    model = network(trackShape, args)
    # `quantize_apply` requires mentioning `DefaultDenseQuantizeConfig` with `quantize_scope`

    # Define the model.
    model.load_weights(args.modelDir)
    with quantize_scope(
      {'DefaultDenseQuantizeConfig': DefaultDenseQuantizeConfig,
       }):
      # Use `quantize_apply` to actually make the model quantization aware.
      quant_aware_model = tfmot.quantization.keras.quantize_apply(model)

    
    if args.verbose == True: quant_aware_model.summary()

    # Compile and train model with early stopping 
    adam = keras.optimizers.Adam(lr = args.learningRate, amsgrad = True)
    earlyStopping = keras.callbacks.EarlyStopping(patience = args.patience)

    
    generatorOptions['batchSize'] = args.batchSize
    generatorOptions['useMultiprocessing'] = args.useMultiprocessing
    # Quantisation doesn't support sigmoid activations (yet) and so set from_logits = True
    quant_aware_model.compile(optimizer = "adam", loss = keras.losses.BinaryCrossentropy(from_logits=True), metrics = ["accuracy"])
    genTrain, genValidation, genTest = createSplitGenerators(args.inputFiles,
                                                                 generatorOptions,
                                                                 shuffle = args.shuffle or args.shuffleChunks,
                                                                 shuffleChunks = args.shuffleChunks)
    qat_history = quant_aware_model.fit_generator(generator = genTrain,
                            validation_data = genValidation,
                            use_multiprocessing = args.useMultiprocessing,
                            workers = args.nWorkers,
                            epochs = args.epochs, verbose = args.verbose)

    y_train = genTrain.getTags()
    y_test = genTest.getTags()

    # Can use the generators for prediction too, but need to ensure that there is no shuffling wrt the above
    y_out_train = quant_aware_model.predict_generator(genTrain)
    y_out_test = quant_aware_model.predict_generator(genTest)
    
    #needed to ensure outputs are in [0,1]
    y_out_train = keras.activations.sigmoid(y_out_train)
    y_out_test = keras.activations.sigmoid(y_out_test)

    rocAUC_train = roc_auc_score(y_train, y_out_train)
    rocAUC_test = roc_auc_score(y_test, y_out_test)

    print(('ROC Train:', rocAUC_train))
    print(('ROC Test:', rocAUC_test))
    return quant_aware_model


def evalModel(args, model, forceCPUTrue):
    #tf-1.X
    #config = tf.ConfigProto(device_count={'GPU': 0})
    #sess = tf.Session(config=config)

    #tf-2.X
    #set CPU as avalible device
    if forceCPUTrue == True:
        #tf.config.set_visible_devices([],"GPU")
        deviceUsed = "CPU"
    else:
        deviceUsed = "GPU"
    generatorOptions['batchSize'] = 1
    generatorOptions['dataset'] = args.sample if not args.evaluation else 'evaluation'

    if args.nEvents : generatorOptions["dataSize"] = args.nEvents
    genEval = DataGenerator(args.inputFiles, **generatorOptions)

    #measure n evaluations of the data set, then return statistics on them
    timeSamples = np.zeros(args.evalRepeats)
    with tf.device("/cpu:0"):
        for i in range(args.evalRepeats):
            start = time.time()
            y_out = model.predict_generator(genEval)
            timeSamples[i] = time.time()-start
    print("Model '{}' took {} +/- {} s to evaluate the data set on {}".format(args.modelName, np.mean(timeSamples), np.std(timeSamples), deviceUsed))

if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-f", "--inputFiles", type = str, dest = "inputFiles", nargs = '+', default = "DTT_MC2015_Reco15aStrip24_DIMUON_Bd2JpsiKstar.h5", help = 'Input file names.')
    argParser.add_argument("-e", "--epochs", type = int, dest = "epochs", default = 1, help = 'Number of epochs to train for.')
    argParser.add_argument("-n", "--name", type = str, dest = "modelName", default = "tag", help = 'Model name.')
    argParser.add_argument("-v", "--verbose", default = False, action = "store_true", dest = "verbose", help = 'Verbose mode on.')
    argParser.add_argument("-b", "--batchSize", type = int, dest = "batchSize", default = 2 ** 14, help = 'Training batch size.')
    argParser.add_argument("-l", "--learningRate", type = float, dest = "learningRate", default = 1E-3, help = 'Adam learning rate, set lower for fine tuning.')
    
    argParser.add_argument("--network", type = str, dest = "network", default = "tagNetwork", help = 'Network model name.')
    argParser.add_argument("--nEvents", type = int, dest = "nEvents", default = None, help = 'Number of total events to evaluate on.')
    argParser.add_argument("--shuffle", default = False, action = "store_true", dest = "shuffle", help = 'Shuffle traning data.')
    argParser.add_argument("--shuffleChunks", default = False, action = "store_true", dest = "shuffleChunks", help = 'Shuffle training data according to chunks.')
    argParser.add_argument("--nCPU", type = int, dest = "nCPU", default = 1, help = 'Number of threads to use.')
    argParser.add_argument("--nWorkers", type = int, dest = "nWorkers", default = 1, help = 'Number of data loaders.')
    argParser.add_argument("--useMultiprocessing", default = False, action = "store_true", dest = "useMultiprocessing", help = 'Use multiprocessing with nWorkers workers.')
    argParser.add_argument("--patience", type = int, dest = "patience", default = 100, help = 'Early stopping patience (epochs).')
    argParser.add_argument("--outputDir", type = str, dest = "outputDir", default = "./", help = 'Directory to store model and plots')
    argParser.add_argument("--nHidden", type = int, dest = "nHidden", default = 16, help = "Number of outputs for hidden dense layers")
    argParser.add_argument("--fineTune", type = bool, dest = "fineTune", default = False, help = "Choose to fine tune an existing model")
    argParser.add_argument("--modelDir", type = str, dest = "modelDir", default = "model.h5", help = "Location of trained model (if fine tuning)")
    argParser.add_argument("--evaluation", default = False, action = "store_true", dest = "evaluation", help = 'Run in evaluation mode over the full input dataset.')
    argParser.add_argument("--evalRepeats", type = int, dest = "evalRepeats", default = 10, help = "Number of times to repeat measurment of evaluation time")
    argParser.add_argument("--sample", type = str, dest = "sample", default = "validation", help = 'Chosen sample: validation (default), test or training')
    
    argParser.add_argument("--kernel1", type = int, dest = "kernel1", default = 10, help = "Parameter to vary")
    argParser.add_argument("--kernel2", type = int, dest = "kernel2", default = 5, help = "Parameter to vary")
    argParser.add_argument("--filter1", type = int, dest = "filter1", default = 25, help = "Parameter to vary")
    argParser.add_argument("--filter2", type = int, dest = "filter2", default = 5, help = "Parameter to vary")
    argParser.add_argument("--quantBits", type = int, dest = "quantBits", default = 8)
    args = argParser.parse_args()

    if args.fineTune == True:
        quant_model, qat_history = quant_aware_tune(args)
    else:
        quant_model, qat_history  = quant_aware_train(args)
    #evalModel(args, quant_model, True)
    export_history_graph(args, qat_history)
