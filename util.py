"""Utility functions for the multilayer neural network.

Author: Chris Yeung
Student number: 20055209
Date modified: 2/2/2022
"""

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def get_confusion_matrix(pred, true):
    """Returns the confusion matrix for the predicted class labels.

    :param pred: an array of shape (n_samples, n_classes) of the prediction
    :param true: an array of shape (n_samples, n_classes) of ground truth
    :return: an array of shape (n_classes, n_classes) containing the
             confusion matrix
    """
    return confusion_matrix(true.argmax(axis=1), pred.argmax(axis=1))


def save_confusion_matrix(cm, save_path, num_classes, phase):
    """Generates and saves a plot of the given confusion matrix.

    :param cm: an array of the confusion matrix
    :param save_path: the directory to save the plot of the confusion matrix
    :param num_classes: the number of class labels
    :param phase: a string to identify the phase in which the confusion
                  matrix was generated
    :return: None
    """
    classes = [str(digit) for digit in range(num_classes)]
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    plt.figure(figsize=(15, 10))
    plt.title("Confusion matrix - " + phase)
    cm_plt = sn.heatmap(df_cm, annot=True, fmt="g")
    cm_plt.set(xlabel="Predicted label", ylabel="True label")
    bottom, top = cm_plt.get_ylim()
    cm_plt.set_ylim(bottom + 0.5, top - 0.5)
    cm_plt.figure.savefig(save_path + "/" + phase + "_confusion_matrix.png")
