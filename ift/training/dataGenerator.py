import numpy as np
import h5py
import os

from itertools import compress

import tensorflow.keras as keras 

from ift.utils.utils import flatten

'''
TODO:

Support ROOT input files (and perform pre-processing and scaling steps on the fly)
Do scaling on the fly with an input scaling function
Cache pre-computed transformations to disk

'''

def combineFiles(fileNames, keys):

    '''
    Use a virtual dataset in a temporary .h5 file to combine files with entries of the same shape,
    so that they appear to be one contiguous dataset.
    '''

    tmpFile = '/tmp/tmpVDS.h5'

    if os.path.exists(tmpFile):
        os.remove(tmpFile)

    for key in keys:

        sources = []
        totalLength = 0
        shape = None

        for fileName in fileNames:
            with h5py.File(fileName, 'r') as tmpF:

                source = h5py.VirtualSource(tmpF[key])

                shape = source.shape[1:]

                totalLength += source.shape[0]

                sources.append(source)

        layout = h5py.VirtualLayout(shape = (totalLength,) + tuple(shape),
                                    dtype = np.float)

        offset = 0
        for source in sources:

            length = source.shape[0]

            layout[offset : offset + length] = source

            offset += length

        with h5py.File(tmpFile, 'a', libver = 'latest') as f:
            f.create_virtual_dataset(key, layout, fillvalue = np.nan)

    return tmpFile

def checkFilesToCombine(fileNames, keys):

    '''
    Check that the files to combine all have the keys required, and all of the datasets
    are of the same shape.
    '''

    shapes = {k : set() for k in keys}

    for name in fileNames:

        with h5py.File(name, 'r') as f:

            keysPresent = list(f.keys())

            for key in keys:
                if key not in keysPresent:
                    raise ValueError('CombineDatasets: Not all keys present in all files.')

                # Shape except the data size dimension
                if isinstance(f[key], h5py.Dataset):
                  shapes[key].add(f[key].shape[1:])
                else:
                  shapes[key].add(0)

    badShapes = list([len(s) > 1 for s in list(shapes.values())])
    badKeys = list(compress(list(range(len(badShapes))), badShapes))

    if len(badKeys) > 0:
        raise ValueError('CombineDatasets: Shape mismatch in ' + str(np.array(list(shapes.keys()))[badKeys]))

def getDataSize(file, key, frac = 1.0):

    '''
    Get the total length (first/'event' axis) of a single dataset.
    '''

    return int(h5py.File(file, 'r')[key].shape[0] * frac)

def getTotalDataSize(files, key):

    '''
    Get the total length (first/'event' axis) of a (combined) dataset.
    '''

    sum = np.sum([getDataSize(name, key) for name in files])

    return sum

def getShuffleChunkIndices(fileName, key, chunkSize = None, frac = 1.0):

    '''
    Get indices of the chunks from a (file, key).
    '''

    file = h5py.File(fileName, 'r')

    chunkSize = file[key].chunks[0] if not chunkSize else chunkSize
    length = int(file[key].shape[0] * frac)

    # Drop the last entries that are not divisible by chunkSize
    indices = np.arange(chunkSize * (length // chunkSize))

    # np.random.shuffle shuffles according to the first dimension, so group these by chunk
    indices = indices.reshape(-1, chunkSize, 1)

    return indices

def calculateIndices(inputFiles, key, shuffle = True, shuffleChunks = True, frac = 1.0):

    '''
    Calculate ((chunk) shuffled) indices over multiple files, to feed to the generators.
    '''

    inputFiles = [inputFiles] if not isinstance(inputFiles, list) else inputFiles

    if not shuffle:
        return np.arange(int(getTotalDataSize(inputFiles, key) * frac))

    if not shuffleChunks:

        size = int(getTotalDataSize(inputFiles, key) * frac)
        indices = list(range(size))
        np.random.shuffle(indices)

        return indices

    else:
        # Ensure no chunks are split between files (otherwise the chunks
        # will be misaligned at the start of a new file) by dropping the remainder
        # in each file, but ofsetting by the total length (including remainder)

        chunkSize = h5py.File(inputFiles[0], 'r')[key].chunks[0]

        offset = 0
        indices = []

        for f in inputFiles:
            indices.append(getShuffleChunkIndices(f, key, chunkSize, frac) + offset)
            offset += getDataSize(f, key, frac)

        indices = np.concatenate(indices)

        # Do the shuffle of the chunks (over all input files), then reshape

        np.random.shuffle(indices)
        indices = indices.reshape(-1, 1).flatten()

        return indices

def createSplitGenerators(inputFiles, config, shuffle = True, shuffleChunks = True, frac = 1.0, returnIndices = False):

    '''

    A wrapper for creating train/test/validation data generators, particularly useful when the data
    needs to be shuffled. Here the shuffling is done consistently by passing the indices of the dataset
    manually to each, to make sure that events aren't re-used between the categories.

    This is particularly important if the events come form different files, as otherwise all of the
    train/test data may came from the same file, which could be different to the others.

    '''

    indices = calculateIndices(inputFiles, config['featureName'], shuffle, shuffleChunks, frac)

    if 'dataSize' in config and config['dataSize']:
        indices = indices[:config['dataSize']]

    dataGeneratorTrain = DataGenerator(inputFiles, dataset = 'train', indices = indices, **config)
    dataGeneratorValidation = DataGenerator(inputFiles, dataset = 'validation', indices = indices, **config)
    dataGeneratorTest = DataGenerator(inputFiles, dataset = 'test', indices = indices, **config)

    if not returnIndices:
        return dataGeneratorTrain, dataGeneratorValidation, dataGeneratorTest
    else:
        return dataGeneratorTrain, dataGeneratorValidation, dataGeneratorTest, indices

class DataGenerator(keras.utils.Sequence):

    '''
    Generates data on the fly for Keras training, so that the whole dataset doesn't have to be loaded
    into memory in one go.

    Based on https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    '''

    def __init__(self, fileName, batchSize = 2 ** 10, dataset = 'train', trainingType = 'category', nFeatures = 18, trainFrac = 0.8,
                 testFrac = 0.1, validationFrac = 0.1, nTracks = 100, shuffle = False, dataSize = None, seed = None, indices = None,
                 nClasses = 4, catName = None, featureName = 'x', tagName = None, maskVal = -999, forCNN = False, useWeights = False,
                 useExtraFeatures = False, useMultiprocessing = False, extrasFileName = 'categoryPredictions.h5',
                 extrasFeatureName = 'categoryPredictions', maskFeature = None, evaluationMode = False):

        # A property depends on this, so put it at the beginning
        self.useMultiprocessing = useMultiprocessing

        self.fileName = fileName
        self.type = trainingType
        self.batchSize = batchSize
        self.dataset = dataset
        self.shuffle = shuffle

        # If we shuffle when training, we want to avoid this when evaluating the
        # performance afterwards, so when done, set this to True.
        self.unshuffleMode = False

        self.nFeatures = nFeatures
        self.nTracks = nTracks
        self.nClasses = nClasses

        self.featureName = featureName
        self.tagName = tagName
        self.catName = catName

        self.forCNN = forCNN

        self.useWeights = useWeights
        self.maskVal = maskVal
        self.useExtraFeatures = useExtraFeatures or self.type.lower() == 'tag_plus_category' \
                                                 or self.type.lower() == 'tag_plus_extra'

        self.trainFrac = trainFrac
        self.testFrac = testFrac
        self.validationFrac = validationFrac

        sumFrac = self.trainFrac + self.testFrac + self.validationFrac
        self.trainFrac /= sumFrac
        self.testFrac /= sumFrac
        self.validationFrac /= sumFrac

        # Index of feature to mask in training
        self.maskFeature = maskFeature

        np.random.seed(seed = seed)

        self.extrasFeatureName = extrasFeatureName

        if not isinstance(fileName, list):
            self.fileName = fileName
        else:

            keys = self.requiredKeys()

            checkFilesToCombine(fileName, keys)

            tmpFileName = combineFiles(fileName, keys)

            self.fileName = tmpFileName

        if not self.useMultiprocessing:
            self.fileInstance = h5py.File(self.fileName, 'r')

        if self.useExtraFeatures:
            self.extrasFile = h5py.File(extrasFileName, 'r') if extrasFileName else self.file

        # Uses a *lot* of RAM ?!
        # self.performInputValidation()

        if indices is None : indices = np.arange(len(self.file[self.featureName]))

        self.totalDataSize = dataSize
        if not dataSize:
            self.totalDataSize = len(indices)

        trainIdx = int(self.totalDataSize * trainFrac)
        testIdx = trainIdx + int(self.totalDataSize * testFrac)

        if dataset.lower() == 'train':
            self.indices = indices[:trainIdx]
        elif dataset.lower() == 'test':
            self.indices = indices[trainIdx:testIdx]
        elif dataset.lower() == 'validation':
            self.indices = indices[testIdx:]
        elif dataset.lower() == 'evaluation':
            # Do not do this if the network has seen this data before!
            print('INFO: Evaluating over the full dataset.')
            self.indices = indices
        else:
            raise ValueError('Dataset type unrecognised (must be either train, test, or validation)')

        self.dataSize = len(self.indices)

        # In a configuration where we just have feature inputs, and no
        # target tags/categories. Check this, and set a flag so that the Keras
        # calls get only the features.
        self.evaluationMode = evaluationMode and self.checkEvaluationMode()

        self.weights = self.getWeights() if (self.useWeights and \
                                            'category' in self.type and \
                                            not self.evaluationMode) else None

    def checkEvaluationMode(self):

        # Only reasonable not to have any targets if we're reading over the
        # whole dataset
        if not self.dataset.lower() == 'evaluation':
            return False

        # If the corresponding target name is None, then it's assumed that it
        # doesn't exist
        if self.type.lower()[:3] == 'tag' and self.tagName is None:
            # Select 'tag', 'tag_plus_category', etc.
            return True
        if (self.type.lower() == 'category' or \
           self.type.lower() == 'category_flat') and self.catName is None:
            return True

        return False

    def requiredKeys(self):
        ''' Determine the required set of input arrays given the task. '''

        keys = [self.featureName]
        if not self.tagName is None : keys += [self.tagName]

        if self.type.lower() == 'category_flat' or self.type.lower() == 'category':
            if not self.catName is None : keys += [self.catName]
        if self.type.lower() == 'tag_plus_extra':
            keys += [self.extrasFeatureName]

        return keys

    @property
    def file(self):
        if not self.useMultiprocessing:
            return self.fileInstance
        else:
            return h5py.File(self.fileName, 'r')

    def performInputValidation(self):

        ''' Validate input file. '''

        # TODO: Validate all inputs, and make this more robust (run it earlier)
        # TODO: Fix weirdly high RAM usage - what's up with that?

        if self.type.lower() == 'category_flat':
            # Throw if mixing flat and not flat inputs (which result in different sized batches)

            featuresFlat = len(self.file[self.featureName][[1, 2, 3]].shape) < 3
            catsFlat = self.file[self.catName][[1, 2, 3]].shape[-1] == 1

            if featuresFlat != catsFlat:
                raise ValueError('Input datasets much be both flat or not-flat (not mixed).')

        if self.type.lower() == 'tag_plus_extra' or self.type.lower() == 'tag_plus_category':

            featureFileShape = self.file[self.featureName][:self.totalDataSize].shape[:2]
            extraFileShape = self.extrasFile[self.extrasFeatureName][:self.totalDataSize].shape[:2]

            # Extra feature can be per-event rather than per-track, maybe have an extra config for this?
            eventFeatures = (extraFileShape[:1] == featureFileShape[:1]) and extraFileShape[-1] == 1

            if not self.useExtraFeatures:
                raise ValueError("Type is 'tag_plus_category', but useExtraFeatures is False.")
            if len(self.file[self.featureName].shape) != 3:
                raise ValueError("Feature input must be 3D with track information (not flattened).")
            if featureFileShape  != extraFileShape and not eventFeatures:
                raise ValueError("Shape mismatch between data ({}) and extra features ({}).".format(
                featureFileShape, extraFileShape))

    def getTotalDataSize(self):

        return len(self.file[self.featureName])

    def getWeights(self):

        ''' Get category weights for all entries. '''

        oldIndices, newIndices = self.getNewIndices(self.indices)

        cats = np.array(self.file[self.catName][oldIndices])[newIndices] # get all categories

        # If there are masked values, set these to a dummy category
        cats[cats == self.maskVal] = self.nClasses - 1

        # If there are some other weird categories, set them to zero
        cats[np.isin(cats, list(range(self.nClasses)), invert = True)] = 0

        length = len(cats)
        classWeights = { c : length / len(cats[cats == c]) for c in range(self.nClasses) }

        return classWeights

    def getTags(self, indices = None):

        ''' Get flavour tag for all entries. '''

        if indices is None: indices = self.indices

        # No need to deal with ordering, align indices with features
        if self.unshuffleMode:
          return self.file[self.tagName][list(sorted(indices))]

        # Have to get these from the HDF5 array as a sorted list...

        oldIndices, newIndices = self.getNewIndices(indices)

        tags = self.file[self.tagName][oldIndices]

        # ...but in case the indices were shuffled, re-index these to match the feature order
        # (Remembering that self.indices are the indices in the .h5 file!)

        return np.array(tags)[newIndices]

    def getNewIndices(self, indices):

        idxPairs = zip(list(indices), list(range(len(indices))))
        idxPairs = np.array(sorted(idxPairs, key = lambda x : x[0]))

        return list(idxPairs[:,0]), list(idxPairs[:,1])

    def getCats(self, indices = None):

        ''' Get categories for all entries. '''

        # Same logic as self.getTags

        if indices is None: indices = self.indices

        oldIndices, newIndices = self.getNewIndices(indices)

        if self.type.lower() == 'category':
            if self.unshuffleMode:
                return self._data_generation_category(list(sorted(indices)))[0]

            categories = self._data_generation_category(oldIndices)[0]
        else:
            if self.unshuffleMode:
                return self._data_generation_category_flat(list(sorted(indices)))[0]

            categories = self._data_generation_category_flat(oldIndices)[0]

        return np.array(categories)[newIndices]

    def getExtraFeatures(self):

        return self._getExtraFeatures(self.indices)

    def __len__(self):

        '''Denotes the number of batches per epoch'''

        return int(np.ceil(self.dataSize / self.batchSize))

    def getItemTraining(self, indices):

        # Combine tuples (independent of number of elements)
        if self.type.lower() == 'category':
            return self._data_generation_features(indices) + self._data_generation_category(indices)
        elif self.type.lower() == 'category_flat':
            return self._data_generation_features_flat(indices) + self._data_generation_category_flat(indices)
        elif self.type.lower() == 'tag':
            return self._data_generation_features(indices) + self._data_generation_tag(indices)
        elif self.type.lower() == 'tag_plus_category':
            return self._data_generation_features_extra(indices) + self._data_generation_tag(indices)
        elif self.type.lower() == 'tag_plus_extra':
            # For an additional model input, or if it can't be concatenated (no track dimension), etc
            return self._data_generation_features_extra_separate(indices) + self._data_generation_tag(indices)
        else:
            raise ValueError('Training type unrecognised.')

    def getItemEvaluation(self, indices):

        # In evaluation mode, only return the feature arrays

        if self.type.lower() == 'category':
            return self._data_generation_features(indices)[0]
        elif self.type.lower() == 'category_flat':
            return self._data_generation_features_flat(indices)[0]
        elif self.type.lower() == 'tag':
            return self._data_generation_features(indices)[0]
        elif self.type.lower() == 'tag_plus_category':
            return self._data_generation_features_extra(indices)[0]
        elif self.type.lower() == 'tag_plus_extra':
            return self._data_generation_features_extra_separate(indices)[0]
        else:
            raise ValueError('Evaluation type unrecognised.')

    def __getitem__(self, index):

        ''' Generate one batch of data. '''

        allIndices = self.indices

        # If we're evaluating, no need to shuffle (for alignment with tags)
        if self.unshuffleMode:
          allIndices.sort()

        # Generate indexes of the batch
        # (Index gets shuffled when shuffle = True in model.fit_generator(...))
        indices = allIndices[index * self.batchSize:(index + 1) * self.batchSize]

        if not self.evaluationMode:
            return self.getItemTraining(indices)
        else:
            # Return only the feature array
            return self.getItemEvaluation(indices)

    def on_epoch_end(self):

        ''' Updates indexes after each epoch. '''

        if self.shuffle == True:
            np.random.shuffle(self.indices)

    def _getExtraFeatures(self, indices):

        ''' Get extra (track wise) features (e.g, track category predictions). '''

        oldIndices, newIndices = self.getNewIndices(indices)

        extraFeatures = np.array(self.extrasFile[self.extrasFeatureName][oldIndices])[newIndices]

        if self.forCNN:
            extraFeatures[extraFeatures == self.maskVal] = 0

        return extraFeatures

    def _data_generation_tag(self, indices):

        ''' Generates tags containing batch_size samples. '''

        return (self.getTags(indices),)

    def _data_generation_category(self, indices):

        ''' Generates categories containing batch_size samples '''

        oldIndices, newIndices = self.getNewIndices(indices)

        cats = np.array(self.file[self.catName][oldIndices])[newIndices]

        # If there are masked values, set these to a dummy category
        cats[cats == self.maskVal] = self.nClasses - 1

        # If there are some other weird categories, set them to zero
        cats[np.isin(cats, list(range(self.nClasses)), invert = True)] = 0

        if not self.useWeights:

            # Maintain output type as tuple
            return (keras.utils.to_categorical(cats, num_classes = self.nClasses),)

        sampleWeights = cats.copy()

        for k in list(self.weights.keys()):
            sampleWeights[sampleWeights == k] = self.weights[k]

        return keras.utils.to_categorical(cats, num_classes = self.nClasses), sampleWeights

    def _data_generation_category_flat(self, indices):

        ''' Generates categories containing batch_size samples, flattened over tracks. '''

        oldIndices, newIndices = self.getNewIndices(indices)

        cats = np.array(self.file[self.catName][oldIndices])[newIndices]

        # If these aren't already saved in a flattened form (nBatch * nTracks, 1)
        if cats.shape[-1] != 1:
            cats = cats.flatten().reshape(-1, 1)
            cats = flatten(cats, self.maskVal)

        # If there are some other weird categories, set them to zero
        cats[np.isin(cats, list(range(self.nClasses)), invert = True)] = 0

        if not self.useWeights:

            # Maintain output type as tuple
            return (keras.utils.to_categorical(cats, num_classes = self.nClasses),)

        sampleWeights = cats.copy()

        for k in list(self.weights.keys()):
            sampleWeights[sampleWeights == k] = self.weights[k]

        return keras.utils.to_categorical(cats, num_classes = self.nClasses), sampleWeights.flatten()

    def _data_generation_features(self, indices):

        ''' Generates features containing batch_size samples. '''

        oldIndices, newIndices = self.getNewIndices(indices)

        # Generate features
        X = np.array(self.file[self.featureName][oldIndices])[newIndices]

        if not self.maskFeature is None:
            X = np.concatenate( (X[:,:,:self.maskFeature], X[:,:,self.maskFeature + 1:]), axis = 2 )

        if self.forCNN:
            X[X == self.maskVal] = 0

        # Maintain output type as tuple
        return (X,)

    def _data_generation_features_extra(self, indices):

        ''' Generates features containing batch_size samples, with additional features. '''

        # Generate features
        X = self._data_generation_features(indices)[0]

        oldIndices, newIndices = self.getNewIndices(indices)

        extraFeatures = self._getExtraFeatures(oldIndices)[newIndices]

        X = np.concatenate((X, extraFeatures), -1)

        # Maintain output type as tuple
        return (X,)

    def _data_generation_features_extra_separate(self, indices):

        'Generates features containing batch_size samples, with additional features separately.'

        # Generate features
        X = self._data_generation_features(indices)[0]

        oldIndices, newIndices = self.getNewIndices(indices)

        extraFeatures = self._getExtraFeatures(oldIndices)[newIndices]

        # Maintain output type as tuple
        return ([X, extraFeatures],)

    def _data_generation_features_flat(self, indices):

        ''' Generates features containing batch_size samples, flattened over tracks. '''

        oldIndices, newIndices = self.getNewIndices(indices)

        # Generate features
        X = self.file[self.featureName][oldIndices][newIndices]

        # If these aren't already saved in a flattened form (nBatch * nTracks, nFeatures)
        if len(X.shape) > 2:
            X = flatten(X.reshape(-1, self.nFeatures), self.maskVal)

        # Maintain output type as tuple
        return (X,)
