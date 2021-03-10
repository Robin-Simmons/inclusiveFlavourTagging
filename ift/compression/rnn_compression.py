"""
Run tests for compressing sequential RNNs with pruning
"""

import argparse
import tempfile

import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from ift.training.dataGenerator import createSplitGenerators

__author__ = "Robin Simmons <rs17751@bristol.ac.uk>"

keras = tf.keras

""" Model definitions. Sequential API appears to be needed for proper function of compression in TF 2.2.0 with tfmot 0.5.0"""
def smallRNN_seq(dense_size = 8, dropout_rate = 0.5, rnn_dropout_rate = 0.5):
    model = keras.Sequential([keras.layers.Dense(dense_size, "relu", input_shape = (100,18), name = "dist_dense"),
    keras.layers.GRU(units = dense_size, recurrent_dropout = rnn_dropout_rate,
                            activation = "relu", name = "gru", return_sequences = False),
    keras.layers.Dense(dense_size, "relu", name = "dense"),
    keras.layers.Dropout(dropout_rate),
    keras.layers.Dense(1, name = "out")])
    return model

def minimalRNN_seq(dense_size = 8, dropout_rate = 0.5, rnn_dropout_rate = 0.5):
    model = keras.Sequential([keras.layers.GRU(units = dense_size, recurrent_dropout = rnn_dropout_rate,
                            activation = "relu", name = "gru", return_sequences = False, input_shape = (100,18)),
    keras.layers.Dropout(dropout_rate),
    keras.layers.Dense(1, name = "out")])
    return model

""" Returns AUC-ROC for a model, given a data set (generator) """
def test_auc_roc(model, gen, label):
    tags = gen.getTags()
    logit_out = model.predict(gen)
    out = keras.activations.sigmoid(logit_out)
    auc_roc = roc_auc_score(tags, out)
    print("{} AUC-ROC: {}".format(label, auc_roc))

def return_pruning_params(sparsity, epochs, sparsity_var, fixed_sparsity):
    if sparsity_var == True:
        return {'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity = fixed_sparsity,
                                                               final_sparsity = final_sparsity,
                                                               begin_step = 0,
                                                               end_step = epochs)}
    else:
        return {'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity = sparsity,
                                                               final_sparsity = fixed_sparsity,
                                                               begin_step = 0,
                                                               end_step = epochs)}
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
'testFrac' : 0.1
}

""" Train RNN model with early stopping and ADAM """
def train_model(args):
    model = globals()["{}_seq".format(args.network)](args.hidden_units, args.dropout, args.rnn_dropout)
    model.summary()
    genTrain, genValidation, genTest = createSplitGenerators(args.inputFiles,
                                                                 generatorOptions,
                                                                 shuffle = False,
                                                                 shuffleChunks = False)
    model.compile(optimizer = keras.optimizers.Adam(lr = 5e-3, amsgrad = True),
                  loss = keras.losses.BinaryCrossentropy(from_logits = True), metrics = ["accuracy"])
    callback = keras.callbacks.EarlyStopping(patience = 50)
    history = model.fit(x = genTrain, validation_data = genValidation, epochs = args.training_epochs, callbacks = callback, verbose = args.verbose)
    test_auc_roc(model, genTrain, "Training")
    test_auc_roc(model, genTest, "Test")

    return model

""" Prune a trained model to a given final sparsity """
def prune(args, model, sparsity):
    
    pruning_params = return_pruning_params(sparsity, args.pruning_epochs, args.sparsity_var, args.fixed_sparsity)

    model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
    genTrain, genValidation, genTest = createSplitGenerators(args.inputFiles,
                                                             generatorOptions,
                                                             shuffle = False,
                                                             shuffleChunks = False)
    model_for_pruning.compile(optimizer = keras.optimizers.Adam(lr = 5e-3, amsgrad = True),
                  loss = keras.losses.BinaryCrossentropy(from_logits = True), metrics = ["accuracy"])

    callback = [tfmot.sparsity.keras.UpdatePruningStep(),
                tfmot.sparsity.keras.PruningSummaries(log_dir = tempfile.mkdtemp())
    ]

    history = model_for_pruning.fit(x = genTrain, validation_data = genValidation, epochs = args.pruning_epochs, callbacks = callback, verbose = args.verbose)

    test_auc_roc(model_for_pruning, genTrain, "{} Training".format(sparsity))
    test_auc_roc(model_for_pruning, genTest, "{} Test".format(sparsity))



if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-f", "--inputFiles", type = str, dest = "inputFiles", nargs = '+', default = "DTT_MC2015_Reco15aStrip24_DIMUON_Bd2JpsiKstar.h5", help = 'Input file names.')
    argParser.add_argument("--hidden_units", type = int, dest = "hidden_units", default = 16)
    argParser.add_argument("--training_epochs", type = int, dest = "training_epochs", default = 50)
    argParser.add_argument("--pruning_epochs", type = int, dest = "pruning_epochs", default = 15)
    argParser.add_argument("--quant_epochs", type = int, dest = "quant_epochs", default = 5)
    argParser.add_argument("--dropout", type = float, dest = "dropout", default = 0.5)
    argParser.add_argument("--rnn_dropout", type = float, dest = "rnn_dropout", default = 0.5)
    argParser.add_argument("--sparsity_var", type = bool, dest = "final_sparsity", default = True, help = """When True, tuple_sparsity sets the final sparsities and fixed_sparsity sets
                                                                                                             inital sparsity. When False, the inverse is true""")
    argParser.add_argument("--fixed_sparsity", type = tuple, dest = "fixed_sparsity", default = 0.5, help = "Float setting inital or final sparsity to test")
    argParser.add_argument("--tuple_sparsity", type = tuple, dest = "tuple_sparsity", default = (0.8, 0.9, 0.95), help = "Tuple setting inital or final sparsities to test")

    argParser.add_argument("--verbose", type = bool, dest = "verbose", default = False)
    argParser.add_argument("--network", type = str, dest = "network", default = "smallRNN", help = "Network to train and prune. Options are smallRNN and minimalRNN")
    args = argParser.parse_args()
    trained_model = train_model(args)
    prune(args, trained_model, 0.8)
    prune(args, trained_model, 0.9)
    prune(args, trained_model, 0.95)
