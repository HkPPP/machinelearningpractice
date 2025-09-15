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

import utils


class MINSTransformer():
    def __init__(self):
        mnist = fetch_openml('mnist_784', as_frame=False)
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
    y_train_5 = (y_train == '5')  # True for all 5s, False for all other digits
    y_test_5 = (y_test == '5')

    # We can pick a classifier and train it 
    # good starter os stochastic gradient descent classifier
    # - Good to handle large datasets
    # - Good to handle instances independently, ont-by-one
    # - Good for online learning
    sgd_clf = SGDClassifier(random_state=utils.RANDOM_SEED)
    print("Fitting on 5")
    sgd_clf.fit(X_train, y_train_5)
    # Predict 5 correctly
    print(f"Predicting on 5: {sgd_clf.predict([some_digit])}")

    # Measuring accuracy using cross validation
    # result is above 95%
    print(f"Cross validation score: {cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring='accuracy')}")

    # Training using dummy classifier 
    # A dummy clf is a clf that classifies data through basic rules instead of learning from data
    # - most_frequent: Predicts the most frequent label, useful for imbalanced datasets.
    # - stratified: Randomly predicts labels based on their distribution in the training data, preserving class percentages.
    # - uniform: Randomly selects a class with equal probability.
    # - constant: Always predicts the same label, which can be specified for fixed output scenarios.
    # Source: https://www.geeksforgeeks.org/machine-learning/ml-dummy-classifiers-using-sklearn/ 
    dummy_clf = DummyClassifier()
    dummy_clf.fit(X_train, y_train_5)
    print(f"Dummy prediction = {any(dummy_clf.predict([some_digit]))}")