import numpy as np
import tqdm, os, glob, json

import shelve

from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score

metric = np.diag(np.array([-1, -1, -1, 1])) # same as ROOT

def dot4v(v1, v2):
    '''

    Perform n four-vector dot products on vectors of shape (n, 4)
    (not a matrix product!)

    '''

    return np.einsum('ij,ij->i', v1, np.dot(v2, metric))

def transform_objects(df, data, max_nobj, sort, SORT_COL):
    '''
    Transform dataset

    Args:
    -----
        df: a dataframe with event-level structure where each event is described by a sequence
            of tracks, jets, muons, etc.
        data: an array of shape (nb_events, nb_particles, nb_features)
        max_nobj: number of particles to cut off at. if >, truncate, else, -999 pad

    Returns:
    --------
        modifies @a data in place. Pads with -999

    '''
    #if train/test ds are obtained by dataframe splitting, get the starting index
    start_idx = df.index[0]

    # i = event number, event = all the variables for that event
    for i, event in tqdm.tqdm(df.iterrows(), total=df.shape[0], desc="Creating stream"):

        # objs = [[pt's], [eta's], ...] of particles for each event
        objs = np.array(
                [v.tolist() for v in event.get_values()],
                dtype='float32'
            )

        #Sort object if required
        if sort:
            objs = objs[:, (np.argsort(event[SORT_COL]))[::-1]]

        objs = np.transpose([x for x in np.transpose(objs) if -99999 not in x])

        # total number of tracks
        nobjs = 0
        if len(objs.shape)>1:
            nobjs = objs.shape[1]

        # take all tracks unless there are more than n_tracks
        i = i - start_idx
        if nobjs > 0:
            data[i, :(min(nobjs, max_nobj)), :] = objs.T[:(min(nobjs, max_nobj)), :]

        # default value for missing tracks
        data[i, (min(nobjs, max_nobj)):, :  ] = -999

def scale(data, features, SCALING_DICT = None):
    '''
    Args:
    -----
    data: a numpy array of shape (nb_events, nb_particles, n_variables)
    features: list of keys to be used for the model
    SCALING_DICT: Dictionary of scaling parameters (can be empty)

    Returns:
    --------
        modifies data in place, writes out scaling dictionary
    '''

    SCALING_DICT = dict() if SCALING_DICT == None else SCALING_DICT

    if len(SCALING_DICT) == 0:
        for v, name in enumerate(features):
            f = data[:, :, v]
            slc = f[f != -999]
            m, s = slc.mean(), slc.std()
            slc -= m
            slc /= s
            data[:, :, v][f != -999] = slc.astype('float32')
            SCALING_DICT[name] = {'mean' : float(m), 'sd' : float(s)}

    else:
        for v, name in enumerate(features):
            f = data[:, :, v]
            slc = f[f != -999]
            m = SCALING_DICT[name]['mean']
            s = SCALING_DICT[name]['sd']
            slc -= m
            slc /= s
            data[:, :, v][f != -999] = slc.astype('float32')

    return SCALING_DICT

def create_stream(dataframe, num_obj, sort_col, SORT = True, SCALE = True, SCALING_DICT = None):
    '''
    Args:
    -----
    dataframe: a pandas.DataFrame
    sort_col: column for sorting
    SORT: whether to perform column sorting
    SCALE: perform variable scaling
    SCALING_DICT: dictionary where to store scaling info
    '''

    n_variables = dataframe.shape[1]
    var_names = list(dataframe.keys())

    data = np.zeros((dataframe.shape[0], num_obj, n_variables), dtype='float32')

    # Xall function to build X
    transform_objects(dataframe, data, num_obj, SORT, sort_col)

    # Scale training sample and return scaling dictionary
    if SCALE:
        SCALING_DICT = scale(data, var_names, SCALING_DICT = SCALING_DICT)

        return data, SCALING_DICT

    return data

def flatten(toFlatten, maskValue = -999):

    '''
        Flatten input array and remove mask values to give a flat array of tracks and track features,
        (nEvents * nTracks, nFeatures)

        Needs to operate on train and test separately, as these will have a different number of
        overall masked tracks
    '''

    # To flatten targets, this needs to be shape (batch, tracks, 1)
    nFeatures = toFlatten.shape[-1]

    # Reshape this to (nEvents * nTracks, nFeatures)
    flat = toFlatten.reshape(toFlatten.shape[0] * toFlatten.shape[1], -1)

    # Suck out those that are not masked
    flat = flat[flat != maskValue]

    # ...then need to reshape this back to (nEvents * nTracksNotMasked, nFeatures)
    flat = flat.reshape(flat.shape[0] // nFeatures, nFeatures)

    return flat

def reshapeOutputVar(flatOutput, inputArray, maskValue = -999):
    '''
        Take the (non-flat) input array and a flat track-level output and reshapes the track-level output, such that it
        can be contatenated to the input array (taking into account the masking).

        Also needs to operate separately on train and tests arrays.
    '''
    lastAxis = flatOutput.shape[-1]

    # Get a feature from the input array to determine where the masks are
    prototypeVar = inputArray[:,:,range(lastAxis)].copy()
    prototypeVarShape = prototypeVar.shape

    # Flatten this
    outputWithMasks = prototypeVar.reshape(prototypeVarShape[0] * prototypeVarShape[1], lastAxis)

    # Replace the 'prototype' feature variables with our feature for each track
    outputWithMasks[outputWithMasks != -maskValue] = flatOutput.flatten()

    # Reshape this so that it can be concatenated
    outputWithMasks = outputWithMasks.reshape(prototypeVarShape[0], prototypeVarShape[1], prototypeVarShape[2])

    return outputWithMasks

def concatOutputVar(flatOutput, inputArray, maskValue = -999):

    '''
        Take the (non-flat) input array and a flat track-level output and reshapes the track-level output, such that it
        can be contatenated to the input array (taking into account the masking), and then does so.

        Also needs to operate separately on train and tests arrays.
    '''

    outputWithMasks = reshapeOutputVar(flatOutput, inputArray, maskValue)

    # Concatenate this with the input array, creating a new feature corresponding to 'flatOutput'
    outputArray = np.concatenate((inputArray, outputWithMasks), -1)

    return outputArray

def decision_and_mistag(output):
    '''
        Takes classifier output (in [0, 1]), and assign a tag of 1 for those with output > 0.5, and a
        tag of -1 for those with output < 0.5. Mistag is assigned by taking the distance from 1 and -1,
        respectively.

        Return numpy arrays of this tag decision and mistag
    '''
    decision = np.where(output > 0.5, 1, -1)

    mistag = output.copy()
    mistag[decision == 1] = 1. - mistag[decision == 1]

    return decision, mistag

def write_to_EPM(output, dfCols = None, trueTag = None, fileName = 'tagsToEPM.root'):

    '''
        Takes classifier output (in [0, 1]), associated true tag associations (as PDG MC IDs) in a Pandas Series or
        DataFrame, along with (DataFrame) other variables to be written to the resulting ROOT ifle.

        Writes a ROOT file that can be imported into Espresso Performance Monitor.
    '''

    try:
        import pandas as pd
        from root_pandas import to_root
    except ImportError:
        print('ERROR: Cannot import from root_pandas - no ROOT files have been written.')
        return

    decisions, mistags = decision_and_mistag(output)

    if type(dfCols) == pd.Series:
        dfCols = dfCols.to_frame()
    elif dfCols is None or type(dfCols) != pd.DataFrame:
        dfCols = pd.DataFrame()

    dfCols['tag'] = decisions.flatten().astype(np.int32) # Short_t
    dfCols['eta'] = mistags.flatten().astype(np.double) # Float_t
    if not trueTag is None:
        dfCols['truth'] = trueTag.flatten().astype(np.int32) # Short_t

    to_root(dfCols, fileName, key = 'tree')

def library_reader(libdir, modellibrary, modelname):

    #Check if libdir exists
    while not os.path.isdir(libdir):
        print('\nDirectory \'{}\' does not exist.'.format(libdir))
        print('Enter the correct directory: ', [d for d in os.listdir('./') if os.path.isdir(d)])
        libdir = input(': ')
    if libdir[-1] != '/': libdir = libdir+'/'

    #Check if modellibrary exists
    while not os.path.isfile(libdir+modellibrary):
        print('\nModel library \'{}\' does not exist.'.format(modellibrary))
        print('Enter an available library:', [s.replace(libdir, '') for s in glob.glob(libdir+'*json')])
        modellibrary = input(': ')

    #Check if modelname exists
    with open(libdir+modellibrary) as modelfile:
        saved_models = json.load(modelfile)

    print('\nExisting trained models:')
    for modname, settings in list(saved_models.items()):
        print('------------------------------------------------------------------------')
        print('{:27}{}'.format('Model name:', modname))
        print('{:27}{}'.format('Description:', settings['modeldescr']))
        print('{:27}{}'.format('N. of tracks:', settings['maxtracks']))
        print('{:27}{}'.format('Trained with MC:', settings['mc']))
        print('{:27}{}'.format('B ID:', settings['B_ID']))
        print('{:27}{} events'.format('Trained with:', settings['trainstop']))
        print('{:27}{}'.format('Training epochs:', settings['epochs']))
        print('{:27}{}'.format('AUC score on test sample:', settings['AUC_on_TrDS']))
    print('------------------------------------------------------------------------')

    if modelname not in saved_models:
        print('\nModel \''+modelname+'\' not in library.')

    while modelname not in saved_models:
        print('Enter an existing model name:')
        modelname = input(': ')

    return libdir, modelname, saved_models[modelname]

def library_checker(libdir, modellibrary, modelname):

    #Check if libdir exists
    if libdir[-1] != '/': libdir = libdir+'/'
    if not os.path.isdir(libdir):
        os.makedirs(libdir)

    #Check if modelname is already in library
    saved_models = dict()
    if os.path.isfile(libdir+modellibrary):

        with open(libdir+modellibrary) as modelfile:
            saved_models = json.load(modelfile)

        while modelname in saved_models:
            print('\nModel name \'{}\' already used.'.format(modelname))
            print('Existing models:', [k.encode("utf-8") for k, _ in list(saved_models.items())])
            new_modelname = input('Enter a new model name: ')
            if len(new_modelname):
                modelname = new_modelname

    return libdir, modelname, saved_models

def evaluateOvRPredictions(true, pred, name = 'Train'):

    '''
        Print the result of the scalar scoring functions over the OvR (one-versus-rest) predictions
    '''

    ovrPredictions = getOvRPredictions(true, pred)

    rocAUCs = evaluateOvRScore(roc_auc_score, ovrPredictions)
    prAUCs = evaluateOvRScore(average_precision_score, ovrPredictions)

    print('ROC ' + name + ':', rocAUCs)
    print('Precision-recall ' + name + ':', prAUCs)

    return [rocAUCs, prAUCs]

def evaluateOvRScore(scoreFunc, ovrPredictions):

    '''
        Evaluate the a scoring function over the OvR (one-versus-rest) predictions

        scoreFunc is a binary scoring function with arguments scoreFunc(true, pred)
    '''

    categories = ovrPredictions.keys()

    rocAUCOvRs = {category : 0 for category in categories}

    for category in categories:

        true, pred = ovrPredictions[category]

        rocAUCOvRs[category] = scoreFunc(true, pred)

    return rocAUCOvRs

def getOvRPredictions(trueCategories, predCategories):

    '''
        Get the OvR (one-versus-rest) predictions for true and predicted track categories.

        These should be passed as one-hot encoded arrays of shape (nSamples, nCategories)
    '''

    categoryIdxs = np.argmax(trueCategories, axis = 1)
    categories = list(map(int, np.unique(categoryIdxs)))

    categoryOvRs = {category : None for category in categories}

    for category in categories:

        trueOvR = np.zeros((trueCategories.shape[0], 2))

        trueOvR[:,0] = trueCategories[:,category]
        trueOvR[:,1] = np.logical_not(trueOvR[:,0]).astype(np.int32)

        predOvR = np.zeros((predCategories.shape[0], 2))

        predOvR[:,0] = predCategories[:,category]
        predOvR[:,1] = 1. - predCategories[:,category]

        categoryOvRs[category] = (trueOvR, predOvR)

    return categoryOvRs

def exportForCalibration(y_out, y_true, outputdir = "./"):

    '''
        Export the tagger output to 'shelf' (pickled) data files
    '''

    decision, mistag = decision_and_mistag(y_out)

    shelfFile = outputdir + 'taggingOutputsForEPM' # .db automatically appended

    with shelve.open(shelfFile) as shelf:
        shelf['decision'] = decision
        shelf['mistag'] = mistag
        shelf['true_tag'] = y_true

def saveModel(model, modelName, outputdir = './'):

    '''
        Save the Keras model, using the default method (which can be loaded using
        keras.models.load_model), or as separate architecture and weight files for use with
        LWTNN in C++.
    '''

    # Save everything
    model.save(outputdir + modelName + '.h5')

    # Save architecture and weights explicitly, for LWTNN
    arch = model.to_json(sort_keys = True, indent = 4)
    with open(outputdir + modelName + '-arch.json', 'w') as arch_file:
        arch_file.write(arch)

    model.save_weights(outputdir + modelName + '-weights.h5')

def populateScalingForLWTNN(variablesList, lwtnnVarFileName, scalingFileName):

    '''
        Read from the generated LWTNN variables file, and fill with the correct variable name
        from the list, populate the scaling parameters from the scaling file, and save.
    '''

    with open(lwtnnVarFileName, 'r') as varFile:
        varsDict = json.load(varFile)
    with open(scalingFileName, 'r') as scaleFile:
        scaling = json.load(scaleFile)

    for i, v in enumerate(variablesList):

        # Only one input node, but can be sequence or flat
        var = varsDict['inputs' if len(varsDict['inputs']) > 0 else 'input_sequences'][0]['variables'][i]

        assert(var['name'] == 'variable_' + str(i))

        var['name'] = v
        var['offset'] = -scaling[v]['mean']
        var['scale'] = 1. / scaling[v]['sd']

    with open(lwtnnVarFileName[:-5] + '_scaled.json', 'w') as varFile:
        json.dump(varsDict, varFile, sort_keys = True, indent = 4)

def createEmptyLWTNNVariables():

    '''
        Create an empty dictionary for the LWTNN variables file.
    '''

    vars = {}
    vars['input_sequences'] = []
    vars['inputs'] = []
    vars['outputs'] = []

    return vars
