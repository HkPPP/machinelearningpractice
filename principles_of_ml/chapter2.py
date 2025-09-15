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


import numpy as np


IMAGES_PATH = Path() / "images" / "end_to_end_project"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)
RANDOM_SEED = 42


class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self  # always return self!

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]


def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def shuffle_and_split_data(data, test_ratio, rng):
    shuffled_indices = rng.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


def column_ratio(X):
    return X[:, [0]] / X[:, [1]]


def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]  # feature names out


def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler(),
    )


# corr_matrix = housing.corr(numeric_only=True)

if __name__ == "__main__":
    housing_full = load_housing_data()

    housing_full["income_cat"] = pd.cut(
        housing_full["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )


    strat_train_set, strat_test_set = train_test_split(
        housing_full,
        test_size=0.2,
        stratify=housing_full["income_cat"],
        random_state=42,
    )

    housing = strat_train_set.copy()
    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()

    imputer = SimpleImputer(strategy="median")
    housing_num = housing.select_dtypes(include=[np.number])

    X = imputer.fit_transform(housing_num)
    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)

    isolation_forest = IsolationForest(random_state=42)
    outlier_pred = isolation_forest.fit_predict(X)

    housing_cat = housing[["ocean_proximity"]]
    cat_encoder = OneHotEncoder()
    housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

    ### Feature Scaling and Transformation
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import TransformedTargetRegressor

    target_scaler = StandardScaler()
    scaled_labels = target_scaler.fit_transform(housing_labels.to_frame())
    some_new_data = housing[["median_income"]].iloc[:5]  # pretend this is new data\

    # model = LinearRegression()
    # model.fit(housing[["median_income"]], scaled_labels)
    # scaled_predictions = model.predict(some_new_data)
    # predictions = target_scaler.inverse_transform(scaled_predictions)

    model = TransformedTargetRegressor(LinearRegression(), transformer=StandardScaler())
    model.fit(housing[["median_income"]], housing_labels)
    predictions = model.predict(some_new_data)

    ### Transformation Pipelines
    num_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("standardize", StandardScaler()),
        ]
    )
    num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

    num_attribs = [
        "longitude",
        "latitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income",
    ]
    cat_attribs = ["ocean_proximity"]

    cat_pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")
    )

    preprocessing = ColumnTransformer(
        [
            ("num", num_pipeline, num_attribs),
            ("cat", cat_pipeline, cat_attribs),
        ]
    )

    log_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(np.log, feature_names_out="one-to-one"),
        StandardScaler(),
    )
    cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1.0, random_state=42)
    default_num_pipeline = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler()
    )
    preprocessing = ColumnTransformer(
        [
            ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
            ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
            ("people_per_house", ratio_pipeline(), ["population", "households"]),
            (
                "log",
                log_pipeline,
                [
                    "total_bedrooms",
                    "total_rooms",
                    "population",
                    "households",
                    "median_income",
                ],
            ),
            ("geo", cluster_simil, ["latitude", "longitude"]),
            ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
        ],
        remainder=default_num_pipeline,
    )  # one column remaining: housing_median_age

    housing_prepared = preprocessing.fit_transform(housing)


    forest_reg = make_pipeline(preprocessing, RandomForestRegressor(random_state=42))
    forest_rmses = -cross_val_score(
        forest_reg,
        housing,
        housing_labels,
        scoring="neg_root_mean_squared_error",
        cv=10,
    )
