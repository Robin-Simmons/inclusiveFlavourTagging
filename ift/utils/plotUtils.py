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

rcParams.update({'figure.autolayout': True})

import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    Taken from https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def makeTrainingPlots(model, lossName = 'loss', accName = 'acc', modelName = 'tag', plotdir = './'):

    """
    Make the standard training/validation loss and accuracy plots as a function of epoch.

    """
    print(model.history.history)
    loss     = model.history.history[lossName]
    val_loss = model.history.history['val_' + lossName]
    acc      = model.history.history[accName]
    val_acc  = model.history.history['val_' + accName]

    fig = plt.figure(figsize = (12,9))
    ax = fig.add_subplot(111)

    plt.plot(loss, lw = 1.0)
    plt.plot(val_loss, lw = 1.0)
    plt.legend(["Train", "Test"], loc = "upper left")
    plt.savefig(plotdir + 'loss-' + modelName + '.pdf')
    plt.clf()
    
    fig = plt.figure(figsize = (12,9))
    ax = fig.add_subplot(111)

    plt.plot(acc, lw = 1.0)
    plt.plot(val_acc, lw = 1.0)
    plt.legend(["Train", "Test"], loc = "upper left")
    plt.savefig(plotdir + 'acc-' + modelName + '.pdf')
    plt.clf()

def makeTrainingPlotsTF2(history, lossName = 'loss', accName = 'accuracy', modelName = 'tag', plotdir = './'):

    """
    Slightly hacky workaround til I figure out how to port model.fit_generator to model.fit
    """
    print(history.history)
    loss     = history.history[lossName]
    val_loss = history.history['val_' + lossName]
    acc      = history.history[accName]
    val_acc  = history.history['val_' + accName]

    fig = plt.figure(figsize = (12,9))
    ax = fig.add_subplot(111)

    plt.plot(loss, lw = 1.0)
    plt.plot(val_loss, lw = 1.0)
    plt.savefig(plotdir + 'loss-' + modelName + '.pdf')
    plt.clf()

    fig = plt.figure(figsize = (12,9))
    ax = fig.add_subplot(111)

    plt.plot(acc, lw = 1.0)
    plt.plot(val_acc, lw = 1.0)
    plt.savefig(plotdir + 'acc-' + modelName + '.pdf')
    plt.clf()

def makeOutputDistPlot(output, name = 'training', plotdir = './'):

    fig = plt.figure(figsize = (12,9))
    ax = fig.add_subplot(111)

    plt.hist(output, bins = int(np.sqrt(len(output))), histtype = 'stepfilled')
    plt.savefig(plotdir + name + '-hist.pdf')
    plt.clf()
