from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt

from scipy.ndimage import shift
import utils

KNN_MODEL_NAME = "knn_minst_c3e"
N_NEIGHBORS = 10

def shift_pixels(image):
    """
    Assuming that all images are 28x28 (784 pixels total).
    Returns a list of 4 shifted images: right, down, left and up
    """
    reshaped = image.reshape(28, 28)

    shift_right = shift(reshaped, [0, 1], cval=0)
    shift_down = shift(reshaped, [1, 0], cval=0)
    shift_left = shift(reshaped, [0, -1], cval=0)
    shift_up = shift(reshaped, [1, 0], cval=0)
    return [shift_right, shift_down, shift_left, shift_up]

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
        minst = utils.load_npy("mnist_784")
    except FileNotFoundError:
        minst = fetch_openml("mnist_784", as_frame=False)
        X = minst.data
        y = minst.target
        utils.dump_npy("mnist_784", X)
    minst = fetch_openml("mnist_784", as_frame=False)
    X = minst.data
    y = minst.target
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    # The MNIST dataset returned by fetch_openml() is actually already split into
    # a training set (the first 60,000 images) and a test set (the last 10,000 images)
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
    

    
    