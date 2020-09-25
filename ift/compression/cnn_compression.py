"""
Run tests for compression sequential CNNs with pruning and quantisation, compatable with TF 2.2.0 and tfmot 0.5.0
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
def cnn_model_seq(dense_size = 8, dropout_rate = 0.5, spatial_dropout_rate = 0.2):
    model = keras.Sequential([keras.layers.Dense(dense_size, "relu", input_shape = (100,18), name = "dist_dense"),
    keras.layers.SpatialDropout1D(spatial_dropout_rate),
    keras.layers.Conv1D(filters = 8, kernel_size = 7,
                            activation = "relu", padding = "same", name = "conv"),
    keras.layers.GlobalMaxPool1D(name = "pool"),
    keras.layers.Dense(dense_size, "relu", name = "dense"),
    keras.layers.Dropout(dropout_rate),
    keras.layers.Dense(1, name = "out")])
    return model

def cnn_model_2D_seq(dense_size = 8, dropout_rate = 0.5, spatial_dropout_rate = 0.2):
    model = keras.Sequential([keras.layers.Reshape(input_shape = (100,18), target_shape = (100,1,18)),
    keras.layers.Dense(dense_size, "relu", input_shape = (100,1,18), name = "dist_dense"),
    keras.layers.SpatialDropout2D(spatial_dropout_rate),
    keras.layers.Conv2D(filters = 8, kernel_size = (7, 1),
                            activation = "relu", padding = "same", name = "conv"),
    keras.layers.GlobalMaxPool2D(name = "pool"),
    keras.layers.Dense(dense_size, "relu", name = "dense"),
    keras.layers.Dropout(dropout_rate),
    keras.layers.Dense(1, name = "out"),
    keras.layers.Reshape((1,))])
    return model

def convert_model_1D_2D(model, args):
    # Expands the dimension of the conv layer so the weights can be copied
    model_2D = cnn_model_2D_seq(dense_size = args.hidden_units, dropout_rate = args.dropout, spatial_dropout_rate = args.spatial_dropout)
    model_2D_layer_list = []
    for layer in model_2D.layers:
        model_2D_layer_list.append(layer.get_config()["name"])
        
    for layer in model.layers:
        layer_name = layer.get_config()["name"]
        if layer.get_weights():
            weights = layer.get_weights()
            if layer_name == "conv":
                weights[0] = np.expand_dims(weights[0], axis = 1)
            target_layer = model_2D.layers[model_2D_layer_list.index(layer_name)]
            target_layer.set_weights(weights)
    return model_2D

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
'testFrac' : 0.1,
"forCNN" : True
}

""" Train CNN model with early stopping and ADAM """
def train_model(args):
    model = cnn_model_seq(args.hidden_units, args.dropout, args.spatial_dropout)
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

    test_auc_roc(model_for_pruning, genTrain, "Pruning {}% Training".format(sparsity))
    test_auc_roc(model_for_pruning, genTest, "Pruning {}% Test".format(sparsity))

""" Quantise a trained model using quantisation aware training """
def qat(args, model):
    genTrain, genValidation, genTest = createSplitGenerators(args.inputFiles,
                                                             generatorOptions,
                                                             shuffle = False,
                                                             shuffleChunks = False)
    model_2D = convert_model_1D_2D(model, args)
    
    qat_model = tfmot.quantization.keras.quantize_model(model_2D)
    qat_model.summary()
    qat_model.compile(optimizer = keras.optimizers.Adam(lr = 1e-4, amsgrad = True),
              loss = keras.losses.BinaryCrossentropy(from_logits = True), metrics = ["accuracy"])

    qat_history = qat_model.fit(x = genTrain, validation_data = genValidation, epochs = args.quant_epochs, verbose = args.verbose)

    test_auc_roc(qat_model, genTrain, "Quantised Training")
    test_auc_roc(qat_model, genTest, "Quantised Test")

if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-f", "--inputFiles", type = str, dest = "inputFiles", nargs = '+', default = "DTT_MC2015_Reco15aStrip24_DIMUON_Bd2JpsiKstar.h5", help = 'Input file names.')
    argParser.add_argument("--hidden_units", type = int, dest = "hidden_units", default = 16)
    argParser.add_argument("--training_epochs", type = int, dest = "training_epochs", default = 1)
    argParser.add_argument("--pruning_epochs", type = int, dest = "pruning_epochs", default = 1)
    argParser.add_argument("--quant_epochs", type = int, dest = "quant_epochs", default = 1)
    argParser.add_argument("--dropout", type = float, dest = "dropout", default = 0.5)
    argParser.add_argument("--spatial_dropout", type = float, dest = "spatial_dropout", default = 0.2)
    argParser.add_argument("--sparsity_var", type = bool, dest = "final_sparsity", default = True, help = """When True, tuple_sparsity sets the final sparsities and fixed_sparsity sets
                                                                                                             inital sparsity. When False, the inverse is true""")
    argParser.add_argument("--fixed_sparsity", type = tuple, dest = "fixed_sparsity", default = 0.5, help = "Float setting inital or final sparsity to test")
    argParser.add_argument("--tuple_sparsity", type = tuple, dest = "tuple_sparsity", default = (0.8, 0.9, 0.95), help = "Tuple setting inital or final sparsities to test")
    argParser.add_argument("--verbose", type = bool, dest = "verbose", default = False)
    args = argParser.parse_args()
    trained_model = train_model(args)
    for sparsity in args.tuple_sparsity:
        prune(args, trained_model, sparsity)
    qat(args, trained_model)
