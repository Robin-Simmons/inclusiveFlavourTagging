Utils
=======

Various utility functions and scripts that are used in various places during the training and evaluation of the inclusive tagger.

[utils.py](utils.py)
---
General utilities for transforming track-level inputs, evaluating model performance, and exporting the model parameters. In particular, contains the functions that sort and pad tracks event-wise, used in [createTrainingFiles/createInputFiles.py](../createTrainingFiles/createInputFiles.py).


[plotUtils.py](plotUtils.py)
---
Utility functions for plotting the performance of trained models.


[fillScalingParamsForLWTNN.py](fillScalingParamsForLWTNN.py)
---
Populate the LWTNN 'variables' JSON file with input features names and scaling transformations applied during creation of the HDF5 file.

Takes as input the LWTNN JSON file and the scaling parameter file saved by [createTrainingFiles/createInputFiles.py](../createTrainingFiles/createInputFiles.py), and outputs the a populated LWTNN JSON variables file, assuming the feature names given in [createTrainingFiles/constants.py](../createTrainingFiles/constants.py).
