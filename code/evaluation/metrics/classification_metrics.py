import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sn
from sklearn.metrics import confusion_matrix
import warnings


def classification_confusion_matrix(ground_truth, predicted, classifier_name, directory):

    plt.clf()
    plt.cla()
    plt.close()

    labels = ["AGN", "PSR", "FAKE"]

    conf_matrix = confusion_matrix(ground_truth, predicted, labels=labels)

    plt.rcParams["figure.figsize"] = (6, 6)

    ax = plt.axes()

    sn.heatmap(conf_matrix, annot=True, cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax, cbar=False)

    ax.set_xlabel("Predicted Source Type")
    ax.set_ylabel("Actual Source Type")

    ax.set_title("Confusion Matrix for {} Source Classifier".format(classifier_name))

    Path(directory + "/classifier_confusion_matrices/").mkdir(parents=True, exist_ok=True)

    plt.savefig(directory + "/classifier_confusion_matrices/{}_classifier_confusion_matrix.png".format(
        classifier_name.lower()))

    plt.close()


def classification_precision_recall(ground_truth, predicted):

    precision = []
    recall = []

    labels = np.unique(ground_truth)

    for source_type in labels:

        mask = (ground_truth == source_type)

        true_positives = np.sum(predicted[mask] == source_type)

        false_positives = np.sum(predicted[np.logical_not(mask)] == source_type)

        false_negatives = np.sum(predicted[mask] != source_type)

        # RAISE A WARNING IF TURE POSITIVE, FLASE POSITIVE OR FALSE NEgATIVE == 0

        if (true_positives != 0) or (false_positives != 0):
            precision.append(true_positives / (true_positives + false_positives))
        else:
            warnings.warn("There were no true positive or false positive detections of {}".format(source_type))

        if (true_positives != 0) or (false_negatives != 0):
            recall.append(true_positives / (true_positives + false_negatives))
        else:
            warnings.warn("There were no true positive or false negative detections of {}".format(source_type))

    return sum(precision) / labels.shape[0], sum(recall) / labels.shape[0]






    # values, counts = np.unique(predicted, axis=0, return_counts=True)
    #
    # pred_dict = dict(zip(values, counts))
    #
    # values, counts = np.unique(ground_truth, axis=0, return_counts=True)
    #
    # actual_dict = dict(zip(values, counts))
    #
    # # AGN precision and recall
    #
    # agn_true_positives = abs(actual_dict["AGN"] - pred_dict["AGN"])
    #
    # agn_false_positives = abs(actual_dict["AGN"])

# REFERENCES

# Annotating Seaborn - https://stackoverflow.com/questions/32723798/how-do-i-add-a-title-and-axis-labels-to-seaborn-
# heatmap
# Clear Plots - https://stackoverflow.com/questions/17106288/how-to-forget-previous-plots-how-can-i-flush-refresh
# Plotting Confusion Matrix - https://stackoverflow.com/questions/35572000/how-can-i-plot-a-confusion-matrix
# Scikit-Learn Documentation - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
# Seaborn Documentation - https://matplotlib.org/stable/users/explain/colors/colormaps.html
