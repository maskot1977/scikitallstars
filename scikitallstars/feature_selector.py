import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class ScoreFeatureSelector:
    def __init__(self):
        self.regressor = RandomForestRegressor()
        self.classifier = RandomForestClassifier()
        self.importances = []
        self.success_cols = []

    def __call__(self, X, Y, threshold=0.1):
        X = pd.DataFrame(X)
        Y = pd.DataFrame(Y).iloc[:, 0]
        if len(list(set(list(Y)))) == 2:
            self.is_regressor = False
            self.model = self.classifier
        else:
            self.is_regressor = True
            self.model = self.regressor
        self.success_cols = []
        for i in range(X.shape[1]):
            try:
                self.model.fit(X.iloc[:, [i]], Y)
                score = self.model.score(X.iloc[:, [i]], Y)
                self.importances.append([i, score])
                if score > threshold:
                    self.success_cols.append(i)
            except:
                continue

        return self.success_cols
