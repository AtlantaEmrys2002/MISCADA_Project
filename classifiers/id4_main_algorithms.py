# import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, f1_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import LogitBoost
from sklearn.svm import SVC


# N.B. Did not include General Additive Model (GAM), as not enough details were provided.

def evaluation(model, testing_features, testing_target, estimators=None):

    # N.B. Using test data for evaluation

    # Used for differentiating between boosted LR and other methods
    if estimators is None:
        # Make predictions on testing data
        predictions = model.predict(testing_features)
    else:
        predictions = LogitBoost.predict_logit_boosting(X_test, estimators)

    # Pulsars are positive and AGN are negative
    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel().tolist()

    print("AGN Test Errors (out of " + str(tn + fn) + "): " + str(fn))
    print("PSR Test Errors (out of " + str(tp + fp) + "): " + str(fp))

    psr_sensitivity = round((tp / (fp + tp)) * 100, 3)

    print("PSR Sensitivity: " + str(psr_sensitivity) + "%")

    # Determine accuracy of classifier
    accuracy = round(accuracy_score(testing_target, predictions) * 100, 3)

    print("Overall Accuracy: " + str(accuracy) + "%")

    # Determine F1 Score
    f1 = round(f1_score(testing_target, predictions) * 100, 2)

    print("F1 Score: " + str(f1))


def lr_backward_classification(training_features, training_target, testing_features, testing_target):

    log_regressor = LogisticRegression(random_state=42, max_iter=1000)

    # We are using backward stepwise selection (for comparison) - k-fold = 10
    sfs_backward = SequentialFeatureSelector(LogisticRegression(random_state=42, max_iter=1000), cv=10,
                                             direction='backward')

    # Create pipeline
    model = make_pipeline(sfs_backward, log_regressor)

    # Fit model to training parameters
    model.fit(training_features, training_target)

    print("-------------")
    print("LR (Backward)")
    print("-------------\n")

    # Evaluate model on testing data
    evaluation(model, testing_features, testing_target)

    print("\n")

    return


def lr_forward_classification(training_features, training_target, testing_features, testing_target):

    log_regressor = LogisticRegression(random_state=42, max_iter=1000)

    # We are using  forward stepwise selection (for comparison) - k-fold = 10
    sfs_forward = SequentialFeatureSelector(LogisticRegression(random_state=42, max_iter=1000), cv=10,
                                            direction='forward')

    # Create pipeline
    model = make_pipeline(sfs_forward, log_regressor)

    # Fit model to training parameters
    model.fit(training_features, training_target)

    print("-------------")
    print("LR (Forward)")
    print("-------------\n")

    # Evaluate model on testing data
    evaluation(model, testing_features, testing_target)

    print("\n")

    return


def decision_tree_classification(training_features, training_target, testing_features, testing_target):

    # Create decision tree classifier
    model = DecisionTreeClassifier(random_state=42)

    # Fit model to training parameters
    model.fit(training_features, training_target)

    print("-------------")
    print("Decision Tree")
    print("-------------\n")

    # Evaluate model on testing data
    evaluation(model, testing_features, testing_target)

    print("\n")

    return


def two_stage_classification(training_features, training_target, testing_features, testing_target):

    decision_tree = DecisionTreeClassifier(random_state=42)

    log_regressor = LogisticRegression(random_state=42, max_iter=1000)

    # Assume majority voting is how these methods are combined
    voting_mechanism = VotingClassifier(estimators=[('dt', decision_tree), ('lr', log_regressor)])

    # Fit model to training parameters
    voting_mechanism.fit(training_features, training_target)

    print("-------------")
    print("Two-Stage")
    print("-------------\n")

    # Evaluate model on testing data
    evaluation(voting_mechanism, testing_features, testing_target)

    print("\n")

    return


def SVM_classification(training_features, training_target, testing_features, testing_target):

    # Create decision tree classifier
    model = SVC(random_state=42)

    # Fit model to training parameters
    model.fit(training_features, training_target)

    print("-------------")
    print("SVM")
    print("-------------\n")

    # Evaluate model on testing data
    evaluation(model, testing_features, testing_target)

    print("\n")

    return


def boosted_lr(training_features, training_target, testing_features, testing_target):

    # Fit model to training parameters - built in k-fold cross validation
    estimators = LogitBoost.fit_logitboosting(training_features, training_target)

    print("-------------")
    print("Boosted Logistic Regression")
    print("-------------\n")

    # Evaluate model on testing data
    evaluation(None, testing_features, testing_target, estimators)

    print("\n")

    return


# AGN vs PULSARS CLASSIFICATION

# Read data
agn_v_pulsars = pd.read_csv('./datasets/agn_and_pulsars.csv')

# TRAIN-TEST SPLIT

# N.B. we are working with supervised methods, so make sure to separate target and features

# Take only variables which will be used to predict - Spectral Index, Variability_Index, Flux_Density,
# Unc_Energy_Flux100, Signif_Curv, hr12, hr23, hr34, hr45
X = (agn_v_pulsars[['Spectral_Index', 'Variability_Index', 'Flux', 'Unc_Energy_Flux100', 'Signif_Curv', 'hr12',
                   'hr23', 'hr34', 'hr45']]).to_numpy()

# Take pulsar to be 1 and agn to be 0
y = (agn_v_pulsars['pulsarness'] == 'Pulsar').to_numpy()

# 30% test data, 70% training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Apply models
lr_backward_classification(X_train, y_train, X_test, y_test)
lr_forward_classification(X_train, y_train, X_test, y_test)
decision_tree_classification(X_train, y_train, X_test, y_test)
two_stage_classification(X_train, y_train, X_test, y_test)
boosted_lr(X_train, y_train, X_test, y_test)

# REFERENCES

# Key References

# Pandas Documentation - https://pandas.pydata.org/docs/
# sklearn Documentation - https://scikit-learn.org/stable/index.html

# Bug Fixes

# Data Representation sklearn - https://apxml.com/courses/getting-started-with-scikit-learn/chapter-1-intro-setup-scikit
# -learn/data-representation
# Iteration Error - https://stackoverflow.com/questions/62658215/convergencewarning-lbfgs-failed-to-converge-status-1-
# stop-total-no-of-iter
# Reading CSV - https://stackoverflow.com/questions/3518778/how-do-i-read-csv-data-into-a-record-array-in-numpy
# Splitting Data - https://medium.com/@whyamit404/understanding-train-test-split-in-pandas-eb1116576c66
# Voting Classifier - https://medium.com/@heyamit10/voting-classifier-using-sklearn-9e57a8c1384b
