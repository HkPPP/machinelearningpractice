from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier


from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVR
from scipy.stats import loguniform, expon
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier



import pandas as pd

import matplotlib.pyplot as plt

from scipy.ndimage import shift
import utils

import numpy as np

KNN_MODEL_NAME = "knn_minst_c3e"
KNN_MODEL_NAME_EXPANDED = "knn_minst_expanded_c3e"
MNIST_784_DATASET_NAME = "mnist_784"
SHIFTED_MNIST_784_DATASET_NAME = "shifted_mnist_784"
RAND_SEARCH_CV_MODEL_NAME = "rand_search_cv_minst_c3e"
RSCV_BEST_MODEL_NAME = "rscv_best_minst_c3e"
GSCV_BEST_MODEL_NAME = "gscv_best_minst_c3e"
TITANIC_KNN_MODEL_NAME = "knn_titanic_c3e"
TITANIC_DT_MODEL_NAME = "dt_titanic_c3e"


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


def KNeighborsClf(
    X_train, y_train, X_test, y_test, model_name, n_neighbors=10, weights="uniform"
):
    try:
        print(f"Loading model {model_name}")
        kn_clf = utils.load_model(model_name)
    except FileNotFoundError:
        print(f"Model {model_name} not found. Fitting on all numbers")
        kn_clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
        kn_clf.fit(X_train, y_train)
        utils.dump_model(kn_clf, model_name)

    print(f"Training accuracy: {kn_clf.score(X_train, y_train)}")
    print(f"Test accuracy: {kn_clf.score(X_test, y_test)}")
    return kn_clf


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
        print(f"Dataset {MNIST_784_DATASET_NAME} not found. Fetching from openml")
        minst = fetch_openml(MNIST_784_DATASET_NAME, as_frame=False)
        X = minst.data
        y = minst.target
        utils.dump_npz(MNIST_784_DATASET_NAME, X, y)

    # The MNIST dataset returned by fetch_openml() is actually already split into
    # a training set (the first 60,000 images) and a test set (the last 10,000 images)
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

    # kn_clf = KNeighborsClf(X_train, y_train, X_test, y_test, KNN_MODEL_NAME)

    # try:
    #     print(f"Loading model {RSCV_BEST_MODEL_NAME}")
    #     best_knn = utils.load_model(RSCV_BEST_MODEL_NAME)
    # except FileNotFoundError:
    #     try:
    #         print(
    #             f"Model {RSCV_BEST_MODEL_NAME} not found. Loading model {RAND_SEARCH_CV_MODEL_NAME}"
    #         )
    #         rand_search = utils.load_model(RAND_SEARCH_CV_MODEL_NAME)
    #     except FileNotFoundError:
    #         print(
    #             f"Model {RAND_SEARCH_CV_MODEL_NAME} not found. Fitting on all numbers"
    #         )
    #         knn = KNeighborsClassifier()

    #         param_grid = {
    #             "n_neighbors": np.arange(2, 9),
    #             "weights": ["uniform", "distance"],
    #             "p": [1, 2],  # Test L1 (Manhattan) and L2 (Euclidean) distance metrics
    #         }
    #         rand_search = RandomizedSearchCV(
    #             estimator=knn,
    #             param_distributions=param_grid,
    #             cv=5,  # Use 5-fold cross-validation
    #             n_jobs=8,
    #             scoring="accuracy",
    #             verbose=3,
    #             random_state=utils.RANDOM_SEED,
    #         )
    #         print("Fitting rand_search")
    #         rand_search.fit(X_train, y_train)
    #         utils.dump_model(rand_search, RAND_SEARCH_CV_MODEL_NAME)

    #     best_knn = rand_search.best_estimator_
    #     utils.dump_model(best_knn, RSCV_BEST_MODEL_NAME)
    # print(f"Best hyperparameters found: {best_knn.best_params_}")
    # print(f"Best cross-validation accuracy: {best_knn.best_score_:.4f}")
    # print(f"Accuracy on test set: {best_knn.score(X_test, y_test):.4f}")

    ################### Problem 1 ###################
    try:
        print(f"Loading model {GSCV_BEST_MODEL_NAME}")
        best_knn = utils.load_model(GSCV_BEST_MODEL_NAME)
    except FileNotFoundError:
        print(f"Model {GSCV_BEST_MODEL_NAME} not found. Fitting on all numbers")
        knn = KNeighborsClassifier()
        param_grid = {
                "n_neighbors": np.arange(2, 9),
                "weights": ["uniform", "distance"],
                "p": [1, 2],  # Test L1 (Manhattan) and L2 (Euclidean) distance metrics
            }
        grid_search = GridSearchCV(
            estimator=knn,
            param_grid=param_grid,
            cv=5,  # Use 5-fold cross-validation
            n_jobs=8,
            scoring="accuracy",
            verbose=3,
        )
        print("Fitting grid_search")
        grid_search.fit(X_train, y_train)

        best_knn = grid_search.best_estimator_
        utils.dump_model(best_knn, GSCV_BEST_MODEL_NAME)

    print(f"Best hyperparameters found: {best_knn.best_params_}")
    print(f"Best cross-validation accuracy: {best_knn.best_score_:.4f}")
    print(f"Accuracy on test set: {best_knn.score(X_test, y_test):.4f}")

    # # # Problem 2
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
    try:
        print(f"Loading model {KNN_MODEL_NAME_EXPANDED}")
        knn_expanded = utils.load_model(KNN_MODEL_NAME_EXPANDED)
    except FileNotFoundError:
        print(f"Model {KNN_MODEL_NAME_EXPANDED} not found. Fitting on all numbers")
        knn_expanded = KNeighborsClassifier(**best_knn.best_params_)
        knn_expanded.fit(X_train_expanded, y_train_expanded)
        utils.dump_model(knn_expanded, KNN_MODEL_NAME_EXPANDED)

    print(f"Training accuracy: {knn_expanded.score(X_train_expanded, y_train_expanded)}")
    print(f"Test accuracy: {knn_expanded.score(X_test_expanded, y_test_expanded)}")

    ############ Problem 3 ###################
    # titanic_train, titanic_test = utils.load_titanic_data()
    # # print(f"Training head: \n{titanic_train.head(20)}")

    # strat_train_set, strat_test_set = train_test_split(
    #     titanic_train,
    #     test_size=0.2,
    #     random_state=utils.RANDOM_SEED,
    # )
    # X_train = strat_train_set.copy()
    # y_train = strat_train_set["Survived"].copy()
    # X_test = strat_test_set.copy()
    # y_test = strat_test_set["Survived"].copy()

    # print(f"Training info: \n{X_train.info()}")
    # print(f"Training head: \n{X_train.head(20)}")
    # print(f"Test info: \n{X_test.info()}")
    # print(f"Test head: \n{X_test.head(20)}")
    # print(X_train.describe())

    # women = X_train.loc[X_train.Sex == "female"]["Survived"]
    # rate_women = sum(women) / len(women)
    # print("% of women who survived:", rate_women)

    # men = X_train.loc[X_train.Sex == "male"]["Survived"]
    # rate_men = sum(men) / len(men)
    # print("% of men who survived:", rate_men)

    # print("Age not NA:\n", X_train.loc[X_train["Age"].notna()])

    # ### Pipeline
    # cat_pipeline = make_pipeline(
    #     SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")
    # )
    # num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    # default_num_pipeline = make_pipeline(
    #     SimpleImputer(strategy="median"), StandardScaler()
    # )

    # ### Collumns
    # collumns_to_drop = ["PassengerId", "Name", "Ticket", "Survived", "Cabin"]
    # one_hot_collums = ["Sex", "Embarked"]
    # impute_collums = ["Age"]

    # ### Preprocessing
    # preprocessing = ColumnTransformer(
    #     [
    #         ("imputer", SimpleImputer(strategy="median"), impute_collums),
    #         ("collum_dropper", "drop", collumns_to_drop),
    #         (
    #             "one_hot_encoder",
    #             OneHotEncoder(handle_unknown="ignore"),
    #             one_hot_collums,
    #         ),
    #     ]
    # )
    # X_train_prep = preprocessing.fit_transform(X_train)
    # print(f"X train {X_train_prep.shape}")
    # X_test_prep = preprocessing.transform(X_test)
    # print(f"X test {X_test_prep.shape}")

    # knc = make_pipeline(
    #     preprocessing,
    #     KNeighborsClassifier(n_neighbors=10, weights="uniform"),
    # )

    # knc.fit(X_train, y_train)
    # print(f"Training accuracy: {knc.score(X_train, y_train)}")

    # print(f"Original KNeighborsClassifier Test accuracy: {knc.score(X_test, y_test)}")

    # try:
    #     print(f"Loading model {TITANIC_KNN_MODEL_NAME}")
    #     knn_clf = utils.load_model(TITANIC_KNN_MODEL_NAME)
    # except FileNotFoundError:
    #     print(f"Model {TITANIC_KNN_MODEL_NAME} not found. Running grid search")
    #     param_grid = {
    #         "n_neighbors": np.arange(1, 15),
    #         "weights": ["uniform", "distance"],
    #         "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
    #         "leaf_size": np.arange(1, 15),
    #         "p": np.arange(1, 3),
    #     }

    #     knn_clf = GridSearchCV(
    #         KNeighborsClassifier(),
    #         param_grid=param_grid,
    #         cv=5,
    #         n_jobs=20,
    #         verbose=3,
    #         scoring="accuracy",
    #     )
    #     knn_clf.fit(X_train_prep, y_train)
    #     utils.dump_model(knn_clf, TITANIC_KNN_MODEL_NAME)

    # print(f"Best params: {knn_clf.best_params_}")
    # print(f"Best score: {knn_clf.best_score_}")

    # print(
    #     f"Grid search KNeighborsClassifier training accuracy: {knn_clf.score(X_train_prep, y_train)}"
    # )
    # print(
    #     f"Grid search KNeighborsClassifier test accuracy: {knn_clf.score(X_test_prep, y_test)}"
    # )

    # try:
    #     print(f"Loading model {TITANIC_DT_MODEL_NAME}")
    #     dtree = utils.load_model(TITANIC_DT_MODEL_NAME)
    # except FileNotFoundError:
    #     print(f"Model {TITANIC_DT_MODEL_NAME} not found. Running grid search")
    #     param_grid = {
    #         "criterion": ["gini", "entropy"],
    #         "max_depth": [3, 5, 7, 10, None],  # None means no limit
    #         "min_samples_split": [2, 5, 10],
    #         "min_samples_leaf": [1, 2, 4],
    #     }
    #     dtree = GridSearchCV(
    #         DecisionTreeClassifier(),
    #         param_grid=param_grid,
    #         cv=5,
    #         n_jobs=20,
    #         verbose=3,
    #         scoring="accuracy",
    #     )
    #     dtree.fit(X_train_prep, y_train)
    #     utils.dump_model(dtree, TITANIC_DT_MODEL_NAME)

    # print(f"Dtree best params: {dtree.best_params_}")
    # print(f"Dtree best score: {dtree.best_score_}")
    # print(
    #     f"DecisionTreeClassifier training accuracy: {dtree.score(X_train_prep, y_train)}"
    # )
    # print(f"DecisionTreeClassifier test accuracy: {dtree.score(X_test_prep, y_test)}")

    # titanic_test_prep = preprocessing.transform(titanic_test)
    # predictions = dtree.predict(titanic_test_prep)
