
import pytest
import pandas as pd
from scikitallstars import allstars, depict

def test_allstars_objective():
    x_train = pd.DataFrame([])
    y_train = pd.DataFrame([])
    objective = allstars.Objective(x_train, y_train)
