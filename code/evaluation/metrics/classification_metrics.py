import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix


def classification_confusion_matrix(ground_truth, predicted, classifier_name, directory):

    labels = ["AGN", "Pulsar", "FAKE"]

    conf_matrix = confusion_matrix(ground_truth, predicted, labels=labels)

    plt.rcParams["figure.figsize"] = (6, 6)

    ax = plt.axes()

    sn.heatmap(conf_matrix, annot=True, cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax, cbar=False)

    ax.set_xlabel("Predicted Source Type")
    ax.set_ylabel("Actual Source Type")

    ax.set_title("Confusion Matrix for {} Source Classifier".format(classifier_name))

    plt.savefig(directory + "/classifier_confusion_matrices/{}_classifier_confusion_matrix.png".format(
        classifier_name.lower()))

    plt.close()


# REFERENCES

# Annotating Seaborn - https://stackoverflow.com/questions/32723798/how-do-i-add-a-title-and-axis-labels-to-seaborn-
# heatmap
# Plotting Confusion Matrix - https://stackoverflow.com/questions/35572000/how-can-i-plot-a-confusion-matrix
# Scikit-Learn Documentation - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
# Seaborn Documentation - https://matplotlib.org/stable/users/explain/colors/colormaps.html
