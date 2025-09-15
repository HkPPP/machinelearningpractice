from sklearn.model_selection import GridSearchCV
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

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler

from sklearn.metrics.pairwise import rbf_kernel

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector, make_column_transformer


from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

import numpy as np
import utils

RANDOM_SEED = 42


def column_ratio(X):
    return X[:, [0]] / X[:, [1]]


def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]  # feature names out


def get_ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler(),
    )


if __name__ == "__main__":
    #  Downloading the data
    housing_full = utils.load_housing_data()

    # Suppose you’ve chatted with some experts who told you
    # that the median income is a very important attribute
    # to predict median housing prices. You may want
    # to ensure that the test set is representative of the various categories of incomes
    # in the whole dataset. Since the median income is a continuous numerical attribute,
    # you first need to create an income category attribute.
    housing_full["income_cat"] = pd.cut(
        housing_full["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    # stratified sampling: the population is divided into homogeneous subgroups called strata,
    # and the right number of instances are sampled from each stratum
    # to guarantee that the test set is representative of the overall population.
    strat_train_set, strat_test_set = train_test_split(
        housing_full,
        test_size=0.2,
        stratify=housing_full["income_cat"],
        random_state=RANDOM_SEED,
    )

    # drop "income_cat" collumn since it is no longer needed
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    # Make a copy instead of working on the data directly
    housing = strat_train_set.copy()
    # Since we're predicting median_house_value, it is a good idea to drop this collumn
    # to prevent data leakage, where the model just memorize the price
    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()

    # # clean the data by filling in missing data with the median values
    # imputer = SimpleImputer(strategy="median")
    # housing_num = housing.select_dtypes(include=[np.number])
    # X = imputer.fit_transform(housing_num)
    # housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)

    # # Drop outliners
    # isolation_forest = IsolationForest(random_state=42)
    # outlier_pred = isolation_forest.fit_predict(X)
    # housing = housing.iloc[outlier_pred == 1]
    # housing_labels = housing_labels.iloc[outlier_pred == 1]

    # # Transform text data into collumns of yes/no of the ocean_proximity feature
    # housing_cat = housing[["ocean_proximity"]]
    # cat_encoder = OneHotEncoder()
    # housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

    ###################### Preprocessing ######################
    log_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(np.log, feature_names_out="one-to-one"),
        StandardScaler(),
    )
    cluster_simil = utils.ClusterSimilarity(
        n_clusters=10, gamma=1.0, random_state=RANDOM_SEED
    )
    default_num_pipeline = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler()
    )
    cat_pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")
    )
    ratio_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler(),
    )
    preprocessing = ColumnTransformer(
        [
            ("bedrooms", get_ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
            ("rooms_per_house", get_ratio_pipeline(), ["total_rooms", "households"]),
            ("people_per_house", get_ratio_pipeline(), ["population", "households"]),
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
    #############################################################################

    ###################### LinearRegression #################################
    # lin_reg = make_pipeline(preprocessing, LinearRegression())
    # lin_reg.fit(housing, housing_labels)

    # housing_predictions = lin_reg.predict(housing)
    # print(housing_predictions[:5].round(-2))
    # print(housing_labels.iloc[:5].values)
    ##################################################################

    ###################### GridSearchCV #################################
    # full_pipeline = Pipeline(
    #     [
    #         ("preprocessing", preprocessing),
    #         ("random_forest", RandomForestRegressor(random_state=RANDOM_SEED)),
    #     ]
    # )
    # param_grid = [
    #     {
    #         "preprocessing__geo__n_clusters": [5, 8, 10],
    #         "random_forest__max_features": [4, 6, 8],
    #     },
    #     {
    #         "preprocessing__geo__n_clusters": [10, 15],
    #         "random_forest__max_features": [6, 8, 10],
    #     },
    # ]
    # grid_search = GridSearchCV(
    #     full_pipeline, param_grid, cv=3, scoring="neg_root_mean_squared_error"
    # )
    # print("fitting")
    # grid_search.fit(housing, housing_labels)
    # print("parsing")
    # cv_res = pd.DataFrame(grid_search.cv_results_)
    # print("sorting")
    # cv_res.sort_values(by="mean_test_score", ascending=False, inplace=True)

    # print(cv_res.head())
    # print(grid_search.best_params_)
    ##################################################################

    ###################### RandomizedSearchCV ############################################
    # full_pipeline = Pipeline(
    #     [
    #         ("preprocessing", preprocessing),
    #         ("random_forest", RandomForestRegressor(random_state=RANDOM_SEED)),
    #     ]
    # )
    # param_distribs = {
    #     "preprocessing__geo__n_clusters": randint(low=3, high=50),
    #     "random_forest__max_features": randint(low=2, high=20),
    # }

    # rnd_search = RandomizedSearchCV(
    #     full_pipeline,
    #     param_distributions=param_distribs,
    #     n_iter=10,
    #     cv=3,
    #     scoring="neg_root_mean_squared_error",
    #     random_state=42,
    # )

    # print("fitting")
    # rnd_search.fit(housing, housing_labels)
    # final_model = rnd_search.best_estimator_  # includes preprocessing
    # feature_importances = final_model["random_forest"].feature_importances_
    # print(feature_importances.round(2))

    # # extra code – displays the random search results
    # cv_res = pd.DataFrame(rnd_search.cv_results_)
    # cv_res.sort_values(by="mean_test_score", ascending=False, inplace=True)
    # cv_res = cv_res[
    #     [
    #         "param_preprocessing__geo__n_clusters",
    #         "param_random_forest__max_features",
    #         "split0_test_score",
    #         "split1_test_score",
    #         "split2_test_score",
    #         "mean_test_score",
    #     ]
    # ]
    # score_cols = ["split0", "split1", "split2", "mean_test_rmse"]
    # cv_res.columns = ["n_clusters", "max_features"] + score_cols
    # cv_res[score_cols] = -cv_res[score_cols].round().astype(np.int64)
    # print(cv_res.head())

    # # extra code – plots a few distributions you can use in randomized search
    # from scipy.stats import randint, uniform, geom, expon

    # xs1 = np.arange(0, 7 + 1)
    # randint_distrib = randint(0, 7 + 1).pmf(xs1)

    # xs2 = np.linspace(0, 7, 500)
    # uniform_distrib = uniform(0, 7).pdf(xs2)

    # xs3 = np.arange(0, 7 + 1)
    # geom_distrib = geom(0.5).pmf(xs3)

    # xs4 = np.linspace(0, 7, 500)
    # expon_distrib = expon(scale=1).pdf(xs4)

    # plt.figure(figsize=(12, 7))

    # plt.subplot(2, 2, 1)
    # plt.bar(xs1, randint_distrib, label="scipy.randint(0, 7 + 1)")
    # plt.ylabel("Probability")
    # plt.legend()
    # plt.axis([-1, 8, 0, 0.2])

    # plt.subplot(2, 2, 2)
    # plt.fill_between(xs2, uniform_distrib, label="scipy.uniform(0, 7)")
    # plt.ylabel("PDF")
    # plt.legend()
    # plt.axis([-1, 8, 0, 0.2])

    # plt.subplot(2, 2, 3)
    # plt.bar(xs3, geom_distrib, label="scipy.geom(0.5)")
    # plt.xlabel("Hyperparameter value")
    # plt.ylabel("Probability")
    # plt.legend()
    # plt.axis([0, 7, 0, 1])

    # plt.subplot(2, 2, 4)
    # plt.fill_between(xs4, expon_distrib, label="scipy.expon(scale=1)")
    # plt.xlabel("Hyperparameter value")
    # plt.ylabel("PDF")
    # plt.legend()
    # plt.axis([0, 7, 0, 1])

    # plt.show()

    ########################################################################################

    ############################### Problem 1 ###############################
    # from sklearn.svm import SVR
    # from scipy.stats import loguniform, expon

    # svr_pipeline = Pipeline([("preprocessing", preprocessing), ("svr", SVR())])

    # param_grid = [
    #     {
    #         "svr__kernel": ["linear"],
    #         "svr__C": [10000.0, 30000.0, 50000.0, 80000.0, 120000.0],
    #     },
    #     {
    #         "svr__kernel": ["rbf"],
    #         "svr__C": [5000.0, 10000.0, 15000.0, 20000.0, 25000.0],
    #         "svr__gamma": [0.01, 0.1, 1.0]
    #     },
    # ]

    # grid_search = GridSearchCV(
    #     svr_pipeline, param_grid, cv=3, scoring="neg_root_mean_squared_error"
    # )
    # print("fitting")
    # grid_search.fit(housing.iloc[:5000], housing_labels.iloc[:5000])
    # svr_grid_search_rmse = -grid_search.best_score_
    # print(svr_grid_search_rmse)
    # print(grid_search.best_params_)
    #############################################################################################

    ############################### Problem 2 ###############################
    # from sklearn.svm import SVR
    # from scipy.stats import loguniform, expon

    # svr_pipeline = Pipeline([("preprocessing", preprocessing), ("svr", SVR())])

    # param_dist = [
    #     {
    #         "svr__kernel": ["linear", "rbf"],
    #         "svr__C": loguniform(1, 100000),
    #         "svr__gamma": expon(0.1, 1)
    #     },
    # ]

    # grid_search = RandomizedSearchCV(
    #     svr_pipeline, param_distributions=param_dist, cv=3, scoring="neg_root_mean_squared_error"
    # )
    # print("fitting")
    # grid_search.fit(housing.iloc[:5000], housing_labels.iloc[:5000])
    # svr_grid_search_rmse = -grid_search.best_score_
    # print(svr_grid_search_rmse)
    # print(grid_search.best_params_)
    #############################################################################################

    ############################### Problem 3 ###############################
    from sklearn.feature_selection import SelectFromModel
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVR
    from scipy.stats import loguniform, expon

    svr_pipeline = Pipeline(
        [
            ("preprocessing", preprocessing),
            (
                "selector",
                SelectFromModel(
                    estimator=RandomForestRegressor(random_state=RANDOM_SEED)
                ),
            ),
            ("svr", SVR()),
        ]
    )

    param_dist = [
        {
            "svr__kernel": ["linear", "rbf"],
            "svr__C": loguniform(1, 100000),
            "svr__gamma": expon(0.1, 1),
        },
    ]

    rand_search = RandomizedSearchCV(
        svr_pipeline,
        param_distributions=param_dist,
        cv=3,
        scoring="neg_root_mean_squared_error",
        random_state=RANDOM_SEED,
    )
    print("fitting")
    rand_search.fit(housing.iloc[:5000], housing_labels.iloc[:5000])
    svr_grid_search_rmse = -rand_search.best_score_
    print(f"RMSE = {svr_grid_search_rmse}")
    print(f"Best params = {rand_search.best_params_}")

    selector_pipeline = Pipeline(
        [
            ("preprocessing", preprocessing),
            (
                "selector",
                SelectFromModel(
                    RandomForestRegressor(random_state=RANDOM_SEED), threshold=0.005
                ),
            ),
            (
                "svr",
                SVR(
                    C=rand_search.best_params_["svr__C"],
                    gamma=rand_search.best_params_["svr__gamma"],
                    kernel=rand_search.best_params_["svr__kernel"],
                ),
            ),
        ]
    )

    selector_rmses = -cross_val_score(
        svr_pipeline,
        housing.iloc[:5000],
        housing_labels.iloc[:5000],
        scoring="neg_root_mean_squared_error",
        cv=3,
    )
    print(f"Cross val = {pd.Series(selector_rmses).describe()}")

#############################################################################################
