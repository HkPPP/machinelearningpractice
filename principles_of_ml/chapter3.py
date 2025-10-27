from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline

from sklearn.cluster import KMeans


from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer

from sklearn.metrics.pairwise import rbf_kernel

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector, make_column_transformer


from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score

from sklearn.datasets import fetch_openml

from sklearn.linear_model import SGDClassifier
from sklearn.dummy import DummyClassifier
import numpy as np

import principles_of_ml.utils_old as utils_old

SGD_MODEL_NAME = "sgd_clf_minst"
DUMMY_MODEL_NAME = "dummy_clf_minst"
SVM_MODEL_NAME = "svm_clf_minst"
OVR_MODEL_NAME = "ovr_clf_minst"
SDG_MULTI_MODEL_NAME = "sgd_multi_clf_minst"

class MINSTransformer:
    def __init__(self):
        mnist = fetch_openml("mnist_784", as_frame=False)
        self.X, self.y = mnist.data, mnist.target

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X


if __name__ == "__main__":
    minst = MINSTransformer()
    X = minst.X
    y = minst.y

    def plot_digit(image_data):
        image = image_data.reshape(28, 28)
        plt.imshow(image, cmap="binary")
        plt.axis("off")

    some_digit = minst.X[0]
    # plot_digit(some_digit)
    # plt.show()

    # The MNIST dataset returned by fetch_openml() is actually already split into
    # a training set (the first 60,000 images) and a test set (the last 10,000 images)
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

    # Train one digit first
    y_train_5 = y_train == "5"  # True for all 5s, False for all other digits
    y_test_5 = y_test == "5"

    # We can pick a classifier and train it
    # good starter is stochastic gradient descent classifier
    # - Good to handle large datasets
    # - Good to handle instances independently, ont-by-one
    # - Good for online learning
    try:
        print(f"Loading model {SGD_MODEL_NAME}")
        sgd_clf = utils_old.load_model(SGD_MODEL_NAME)
    except FileNotFoundError:
        print("Model {SDG_MODEL_NAME} not found.Fitting on 5")
        sgd_clf = SGDClassifier(random_state=utils_old.RANDOM_SEED)
        sgd_clf.fit(X_train, y_train_5)
        utils_old.dump_model(sgd_clf, SGD_MODEL_NAME)

    # Predict 5 correctly
    print(f"Predicting on 5: {sgd_clf.predict([some_digit])}")

    # Measuring accuracy using cross validation
    # result is above 95%
    print(
        f"Cross validation score: {cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring='accuracy')}"
    )

    # Training using dummy classifier
    # A dummy clf is a clf that classifies data through basic rules instead of learning from data
    # - most_frequent: Predicts the most frequent label, useful for imbalanced datasets.
    # - stratified: Randomly predicts labels based on their distribution in the training data, preserving class percentages.
    # - uniform: Randomly selects a class with equal probability.
    # - constant: Always predicts the same label, which can be specified for fixed output scenarios.
    # Source: https://www.geeksforgeeks.org/machine-learning/ml-dummy-classifiers-using-sklearn/
    try:
        print(f"Loading model {DUMMY_MODEL_NAME}")
        dummy_clf = utils_old.load_model(DUMMY_MODEL_NAME)
    except FileNotFoundError:
        print("Model {DUMMY_MODEL_NAME} not found. Fitting dummy classifier on 5")
        dummy_clf = DummyClassifier()
        dummy_clf.fit(X_train, y_train_5)
        utils_old.dump_model(dummy_clf, DUMMY_MODEL_NAME)
    print(f"Dummy prediction = {any(dummy_clf.predict(X_train))}")

    # To compute the confusion matrix, or to compute the precision and recall,
    # you first need to have a set of predictions
    # so that they can be compared to the actual targets.
    from sklearn.model_selection import cross_val_predict

    y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
    print(f"Cross validation score (without accuracy scoring): {y_train_pred}")

    # Confusion matrix is a matrix that counts the number of instances of class A are classified as class B
    # It shows the number of true positives (TP), true negatives (TN),
    # false positives (FP), and false negatives (FN)
    from sklearn.metrics import confusion_matrix

    print(f"y_train_5 shape: {y_train_5.shape}")
    print(f"y_train_pred shape: {y_train_pred.shape}")
    cm = confusion_matrix(y_train_5, y_train_pred)
    print(
        f"Confusion matrix:\n \
            True Negative: {cm[0][0]}\n \
                False Negative: {cm[0][1]}\n \
                    False Positive: {cm[1][0]}\n \
                        True Positive: {cm[1][1]} \n"
    )

    # Precision is the ratio of true positives to the sum of true positives and false positives
    # Recall is the ratio of true positives to the sum of true positives and false negatives
    # Precision is good in cases where you want to minimize false positives
    # Recall is good when you don't want to take chances and 
    from sklearn.metrics import precision_score, recall_score
    print(f"Precision: {precision_score(y_train_5, y_train_pred)}")
    print(f"Recall: {recall_score(y_train_5, y_train_pred)}")

    from sklearn.svm import SVC
    try:
        print(f"Loading model {SVM_MODEL_NAME}")
        svm_clf = utils_old.load_model(SVM_MODEL_NAME)
    except FileNotFoundError:
        print("Model {SVM_MODEL_NAME} not found. Fitting on all numbers") 
        svm_clf = SVC(random_state=utils_old.RANDOM_SEED)
        svm_clf.fit(X_train[:5000], y_train[:5000])
        utils_old.dump_model(svm_clf, SVM_MODEL_NAME)
    some_digit_scores = svm_clf.decision_function([some_digit])
    print(f"Predicting on 5: {svm_clf.predict([some_digit])}") 
    print(f"Training scores: {some_digit_scores.round(2)}")

    from sklearn.multiclass import OneVsRestClassifier
    try:
        print(f"Loading model {OVR_MODEL_NAME}")
        ovr_clf = utils_old.load_model(OVR_MODEL_NAME)
    except FileNotFoundError:  
        print("Model {OVR_MODEL_NAME} not found. Fitting on all numbers")
        ovr_clf = OneVsRestClassifier(SVC(random_state=utils_old.RANDOM_SEED))
        ovr_clf.fit(X_train[:5000], y_train[:5000])
        utils_old.dump_model(ovr_clf, OVR_MODEL_NAME)

    some_digit_scores = ovr_clf.decision_function([some_digit])
    print(f"Predicting on 5: {ovr_clf.predict([some_digit])}") 
    print(f"Training scores: {some_digit_scores.round(2)}")


    try:
        print("Loading model {SDG_MULTI_MODEL_NAME}")
        sgd_clf = utils_old.load_model(SDG_MULTI_MODEL_NAME)
    except FileNotFoundError:
        print("Model {SDG_MULTI_MODEL_NAME} not found. Fitting on all numbers")
        sgd_clf = SGDClassifier(random_state=utils_old.RANDOM_SEED)
        sgd_clf.fit(X_train[:5000], y_train[:5000])
        utils_old.dump_model(sgd_clf, SDG_MULTI_MODEL_NAME)

    some_digit_scores = sgd_clf.decision_function([some_digit])
    print(f"Predicting on 5: {sgd_clf.predict([some_digit])}") 
    print(f"Training scores: {some_digit_scores.round(2)}")