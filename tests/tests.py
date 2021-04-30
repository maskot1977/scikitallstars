
import sys
import os

sys.path.append(os.path.abspath("../scikitallstars/"))

import pytest
import pandas as pd
from scikitallstars import allstars, depict
import sklearn.datasets
from sklearn.model_selection import train_test_split


def test_allstars_classification():
    dataset = sklearn.datasets.load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=.4) 
    allstars_model = allstars.fit(X_train, y_train, timeout=100, n_trials=10, feature_selection=True)


def test_allstars_regression():
    dataset = sklearn.datasets.load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=.4) 
    allstars_model = allstars.fit(X_train, y_train, timeout=100, n_trials=10, feature_selection=True)

