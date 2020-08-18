"""
Evaluate a saved track category network on a dataset, and save these outputs to a file with
the same shape (with padding) as the inputs.
"""

__author__ = "Daniel O'Hanlon <daniel.ohanlon@cern.ch>"

# Loop through chunks of chunkSize of input file
# Evaluate model using predict_on_batch using chunkSize of dataGenerator
# Transform outputs to the shape (+ padding) determined on the input featureArray form the file
# Create/append to new h5 file with this shape
# (NB these are not scaled - see what happens when they are!)

import os
import h5py
import argparse
import datetime

import hashlib
import subprocess as sp

from tqdm import tqdm

from keras.models import load_model

from ift.utils.utils import reshapeOutputVar
from ift.training.dataGenerator import DataGenerator

def gitRevHash():
    return sp.check_output(['git', 'rev-parse', 'HEAD'])

def hashFile(fileName):

    hasher = hashlib.md5()

    with open(fileName, 'rb') as f:
        b = f.read()
        hasher.update(b)

    return hasher.hexdigest()

def populateMetadata(f, args, genOpts):

    metadata = vars(args)

    for k, v in metadata.items():
        f.attrs[k] = str(v)

    for k, v in genOpts.items():
        f.attrs[k] = v

    f.attrs['date'] = str(datetime.datetime.now())
    f.attrs['gitRevHash'] = gitRevHash().decode("utf-8").strip('\n')

    f.attrs['inputFileHash'] = hashFile(args.inputFile)
    f.attrs['catModelFileHash'] = hashFile(args.modelFile)

def writeChunk(iChunk, chunkSize, dataGenerator, featureArray, outputFile, model):

    h5chunks = (featureArray.chunks[0], featureArray.chunks[1], dataGenerator.nClasses)
    dataSize = dataGenerator.totalDataSize

    minIndex = iChunk * chunkSize

    if minIndex > dataSize:
        return

    maxIndex = min((iChunk + 1) * chunkSize, dataSize)

    indices = list(range(iChunk * chunkSize, maxIndex))
    featureBatch = dataGenerator._data_generation_features_flat(indices)[0]

    predCats = model.predict_on_batch(featureBatch)

    predCatsByEvt = reshapeOutputVar(predCats, featureArray[indices])

    with h5py.File(outputFile, 'a') as output:

        if iChunk == 0:

            maxShape = (None, predCatsByEvt.shape[1], predCatsByEvt.shape[2])
            output.create_dataset('categoryPredictions', data = predCatsByEvt,
                                                         chunks = h5chunks,
                                                         compression = 'lzf',
                                                         maxshape = maxShape)

        else:
            # Append

            newShape = output['categoryPredictions'].shape[0] + predCatsByEvt.shape[0]
            output['categoryPredictions'].resize(newShape, axis = 0)
            output['categoryPredictions'][-predCatsByEvt.shape[0]:] = predCatsByEvt

    return

def writeCategoryPredictions(args, outputFile, generatorOptions):

    model = load_model(args.modelFile)

    dataSize = h5py.File(args.inputFile)['featureArray'].shape[0]

    if args.chunkSize > dataSize:
        raise ValueError('Chunk size is larger than the entire dataset!')

    dataGenerator = DataGenerator(args.inputFile, dataset = 'evaluation', **generatorOptions)
    featureArray = h5py.File(args.inputFile, 'r')['featureArray']

    nChunks = (dataSize // args.chunkSize) + 1

    for iChunk in tqdm(list(range(nChunks))):

        writeChunk(iChunk, args.chunkSize, dataGenerator, featureArray, outputFile, model)

    with h5py.File(outputFile, 'a') as output:

        populateMetadata(output, args, generatorOptions)

if __name__ == '__main__':

    argParser = argparse.ArgumentParser()

    argParser.add_argument("-s", "--chunkSize", type = int, dest = "chunkSize", default = 50000, help = 'Size of chunk to write to file.')
    argParser.add_argument("-b", "--batchSize", type = int, dest = "batchSize", default = 2 ** 14, help = 'Size of batch for generator.')
    argParser.add_argument("-f", "--nFeatures", type = int, dest = "nFeatures", default = 18, help = 'Number of training features.')
    argParser.add_argument("-c", "--nTrackCategories", type = int, dest = "nTrackCategories", default = 4, help = 'Number of track output categories.')
    argParser.add_argument("--nEvents", type = int, dest = "nEvents", default = None, help = 'Number of total events to evaluate on.')

    argParser.add_argument("--modelFile", type = str, dest = "modelFile", default = 'testModel.h5', help = 'Keras model file.')
    argParser.add_argument("--inputFile", type = str, dest = "inputFile", default = 'DTT_MC2015_Reco15aStrip24_DIMUON_Bd2JpsiKstar.h5', help = 'Input data file to run model over.')

    args = argParser.parse_args()

    # Also save mode params, etc, as attributes?
    outputFile = args.inputFile.split('/')[-1][:-3] + '_CategoryPredictions.h5'

    if os.path.exists(outputFile):
        os.remove(outputFile)

    generatorOptions = {

    'trainingType' : 'category',
    'featureName' : 'featureArray', # Ensure that this is not flat, so that the indices align!
    'tagName' : 'tagArray',
    'catName' : 'catArray',

    'nClasses' : args.nTrackCategories,
    'nFeatures' : args.nFeatures,

    'batchSize' : args.batchSize,

    'dataSize' : args.nEvents if args.nEvents else h5py.File(args.inputFile)['featureArray'].shape[0]

    }

    writeCategoryPredictions(args, outputFile, generatorOptions)
