from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt

from scipy.ndimage import shift
import utils

import numpy as np

KNN_MODEL_NAME = "knn_minst_c3e"
N_NEIGHBORS = 10
MNIST_784_DATASET_NAME = "mnist_784"
SHIFTED_MNIST_784_DATASET_NAME = "shifted_mnist_784"


def shift_pixels(image):
    """
    Assuming that all images are 28x28 (784 pixels total).
    Returns a list of 4 shifted images: right, down, left and up
    """
    reshaped = image.reshape(28, 28)

    shift_right = shift(reshaped, [0, 1], cval=0).ravel()
    shift_down = shift(reshaped, [1, 0], cval=0).ravel()
    shift_left = shift(reshaped, [0, -1], cval=0).ravel()
    shift_up = shift(reshaped, [1, 0], cval=0).ravel()
    return [shift_right, shift_down, shift_left, shift_up]


def get_shifted_images_and_labels(X, y):
    X_expanded = []
    y_expanded = []
    for image, label in zip(X, y):
        X_expanded.extend(shift_pixels(image))
        y_expanded.extend([label] * 4)
    X_expanded = np.array(X_expanded)
    y_expanded = np.array(y_expanded)
    return X_expanded, y_expanded


def excercise(X_train, y_train, X_test, y_test):
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    try:
        print(f"Loading model {KNN_MODEL_NAME}")
        kn_clf = utils.load_model(KNN_MODEL_NAME)
    except FileNotFoundError:
        print("Model {KNN_MODEL_NAME} not found. Fitting on all numbers")
        kn_clf = KNeighborsClassifier(n_neighbors=N_NEIGHBORS)
        kn_clf.fit(X_train, y_train)
        utils.dump_model(kn_clf, KNN_MODEL_NAME)

    print(f"Training accuracy: {kn_clf.score(X_train, y_train)}")
    print(f"Test accuracy: {kn_clf.score(X_test, y_test)}")


def plot_digit(image_data):
    plt.imshow(image_data, cmap="binary")
    plt.axis("off")


if __name__ == "__main__":
    try:
        print(f"Loading dataset {MNIST_784_DATASET_NAME}")
        minst = utils.load_npz(MNIST_784_DATASET_NAME)
        X = minst["arr_0"]
        y = minst["arr_1"]
    except FileNotFoundError:
        print("Dataset {MNIST_784_DATASET_NAME} not found. Fetching from openml")
        minst = fetch_openml(MNIST_784_DATASET_NAME, as_frame=False)
        X = minst.data
        y = minst.target
        utils.dump_npz(MNIST_784_DATASET_NAME, X, y)

    # The MNIST dataset returned by fetch_openml() is actually already split into
    # a training set (the first 60,000 images) and a test set (the last 10,000 images)
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

    try:
        print(f"Loading model {SHIFTED_MNIST_784_DATASET_NAME}")
        data = utils.load_npz(SHIFTED_MNIST_784_DATASET_NAME)
        X_train_expanded = data["arr_0"]
        y_train_expanded = data["arr_1"]
        X_test_expanded = data["arr_2"]
        y_test_expanded = data["arr_3"]
    except FileNotFoundError:
        print(
            f"Dataset {SHIFTED_MNIST_784_DATASET_NAME} not found. Creating expanded dataset"
        )
        X_extra, y_extra = get_shifted_images_and_labels(X_train, y_train)
        X_train_expanded = np.concatenate((X_train, X_extra), axis=0)
        y_train_expanded = np.concatenate((y_train, y_extra), axis=0)

        X_extra, y_extra = get_shifted_images_and_labels(X_test, y_test)
        X_test_expanded = np.concatenate((X_test, X_extra), axis=0)
        y_test_expanded = np.concatenate((y_test, y_extra), axis=0)
        utils.dump_npz(
            SHIFTED_MNIST_784_DATASET_NAME,
            X_train_expanded,
            y_train_expanded,
            X_test_expanded,
            y_test_expanded,
        )

    print(f"X_train_expanded shape: {X_train_expanded.shape}")
    print(f"y_train_expanded shape: {y_train_expanded.shape}")
    print(f"X_test_expanded shape: {X_test_expanded.shape}")
    print(f"y_test_expanded shape: {y_test_expanded.shape}")

    # excercise(X_train, y_train, X_test, y_test)
