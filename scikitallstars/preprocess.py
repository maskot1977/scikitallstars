import numpy as np


def remove_low_variance_features(df, threshold=0.0):
    ok_id = []
    for colid, col in enumerate(df.values.T):
        try:
            if np.var(col) > threshold:
                ok_id.append(colid)
        except:
            pass

    return df.iloc[:, ok_id]


def remove_high_correlation_features(df, threshold=0.95):
    corrcoef = np.corrcoef(df.T.values.tolist())
    selected_or_not = {}
    for i, array in enumerate(corrcoef):
        if i not in selected_or_not.keys():
            selected_or_not[i] = True
        if selected_or_not[i]:
            for j, ary in enumerate(array):
                if i < j:
                    if abs(ary) >= threshold:
                        selected_or_not[j] = False

    return df.iloc[:, [i for i, array in enumerate(corrcoef) if selected_or_not[i]]]

import pandas as pd
from sklearn.ensemble import RandomForestRegressor


class TableCleaner:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=1, max_depth=1, n_jobs=-1)
        self.success_col = None
        self.success_row = None

    def clean_columns(self, X, Y):
        X = pd.DataFrame(X)
        Y = pd.DataFrame(Y)
        self.success_col = []

        cols = [i for i in range(X.shape[1])]
        waiting = [cols]
        while len(waiting) > 0:
            cols = waiting.pop()
            try:
                self.model.fit(X.iloc[:, cols], Y.values.ravel())
                self.success_col += cols
            except:
                if len(cols) > 2:
                    waiting.append(cols[: int(len(cols) / 2)])
                    waiting.append(cols[int(len(cols) / 2) :])

        return self.success_col

    def clean_rows(self, X, Y=None):
        X = pd.DataFrame(X)
        self.success_row = [
            i for i, b in enumerate(list(X.isnull().any(axis=1))) if not b
        ]
        return self.success_row
