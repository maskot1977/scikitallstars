import os
import sys

sys.path.append(os.path.abspath("../scikitallstars/"))

import pandas as pd
import pytest
import sklearn.datasets
from sklearn.model_selection import train_test_split

from scikitallstars import allstars, depict


def test_allstars_classification():
    common_process(sklearn.datasets.load_breast_cancer())


def test_allstars_regression():
    common_process(sklearn.datasets.load_diabetes())


def common_process(dataset):
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.4
    )
    allstars_model = allstars.fit(
        X_train,
        y_train,
        timeout=100,
        n_trials=10,
        feature_selection=True,
        show_progress_bar=False,
    )
    depict.feature_importances(allstars_model)
    depict.training_summary(allstars_model)
    depict.best_scores(allstars_model)
    depict.all_metrics(allstars_model, X_train, y_train)
    depict.all_metrics(allstars_model, X_test, y_test)
    allstars_model.score(X_train, y_train), allstars_model.score(X_test, y_test)
    depict.metrics(allstars_model, X_train, y_train, X_test, y_test)
    allstars_model.predict(X_test)
    stacking_model = allstars.get_best_stacking(
        allstars_model,
        X_train,
        y_train,
        timeout=100,
        n_trials=10,
        show_progress_bar=False,
    )
    stacking_model.score(X_train, y_train), stacking_model.score(X_test, y_test)
    depict.metrics(stacking_model, X_train, y_train, X_test, y_test)
    depict.model_importances(stacking_model)
    stacking_model.predict(X_test)


def main():
    test_allstars_classification()
    test_allstars_regression()


if __name__ == "__main__":
    main()
