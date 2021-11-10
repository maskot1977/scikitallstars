from sklearn.base import BaseEstimator, TransformerMixin
from scikitallstars.timeout import on_timeout, handler_func
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
)


class Classifier:
    def __init__(self, params, debug=False):
        self.params = params
        self.debug = debug
        if params["standardize"] == "StandardScaler":
            self.standardizer = StandardScaler()
        elif params["standardize"] == "MinMaxScaler":
            self.standardizer = MinMaxScaler()
        elif params["standardize"] == "NoScaler":
            self.standardizer = NullScaler()

        if params["model_name"] == "RandomForest":
            self.model = RandomForestClassifier(**params["model_params"])
        elif params["model_name"] == "SVC":
            self.model = SVC(**params["model_params"])
        elif params["model_name"] == "MLP":
            self.model = MLPClassifier(**params["model_params"])
        elif params["model_name"] == "LogisticRegression":
            self.model = LogisticRegression(**params["model_params"])
        elif params["model_name"] == "GradientBoosting":
            self.model = GradientBoostingClassifier(**params["model_params"])
        elif params["model_name"] == "kNN":
            self.model = KNeighborsClassifier(**params["model_params"])
        elif params["model_name"] == "Ridge":
            self.model = RidgeClassifier(**params["model_params"])
        elif params["model_name"] == "LDA":
            self.model = LinearDiscriminantAnalysis(**params["model_params"])
        elif params["model_name"] == "QDA":
            self.model = QuadraticDiscriminantAnalysis(**params["model_params"])
        elif params["model_name"] == "ExtraTrees":
            self.model = ExtraTreesClassifier(**params["model_params"])
        elif params["model_name"] == "AdaBoost":
            self.model = AdaBoostClassifier(**params["model_params"])
        if self.debug:
            print(self.model)

    def _fit_and_predict_core(
        self, x, y=None, fitting=False, proba=False, support=None, score=False
    ):
        if support is None:
            if fitting == True:
                self.standardizer.fit(x)

            self.standardizer.transform(x)
            if score:
                pred = np.array(self.model.predict(x))
                return f1_score(pred.flatten(), np.array(y).flatten())

            if fitting == True:
                self.model.fit(x, y)
            if y is None:
                if proba:
                    return self.model.predict_proba(x)
                else:
                    return self.model.predict(x)
        else:
            if fitting == True:
                self.standardizer.fit(x.iloc[:, support])

            self.standardizer.transform(x.iloc[:, support])
            if score:
                pred = np.array(self.model.predict(x.iloc[:, support]))
                return f1_score(pred.flatten(), np.array(y).flatten())

            if fitting == True:
                self.model.fit(x.iloc[:, support], y)

            if y is None:
                if proba and hasattr(self.model, "predict_proba"):
                    return self.model.predict_proba(x.iloc[:, support])
                else:
                    return self.model.predict(x.iloc[:, support])

        return None

    @on_timeout(limit=600, handler=handler_func, hint=u"classifier.fit")
    def fit(self, x, y, support=None):
        self._fit_and_predict_core(x, y, fitting=True, support=support)
        return self

    def predict(self, x, support=None):
        pred_y = self._fit_and_predict_core(x, support=support)
        return pred_y

    def predict_proba(self, x, support=None):
        pred_y = self._fit_and_predict_core(x, proba=True, support=support)
        return pred_y

    def score(self, x, y, support=None):
        return self._fit_and_predict_core(x, y, support=support, score=True)


class Regressor:
    def __init__(self, params, debug=False, support=None):
        self.params = params
        self.debug = debug
        self.support = support
        if params["standardize"] == "StandardScaler":
            self.standardizer = StandardScaler()
        elif params["standardize"] == "MinMaxScaler":
            self.standardizer = MinMaxScaler()
        elif params["standardize"] == "NoScaler":
            self.standardizer = NullScaler()

        if params["model_name"] == "RandomForest":
            self.model = RandomForestRegressor(**params["model_params"])
        elif params["model_name"] == "SVR":
            self.model = SVR(**params["model_params"])
        elif params["model_name"] == "MLP":
            self.model = MLPRegressor(**params["model_params"])
        elif params["model_name"] == "LinearRegression":
            self.model = LinearRegression(**params["model_params"])
        elif params["model_name"] == "PLS":
            self.model = PLSRegression(**params["model_params"])
        elif params["model_name"] == "GradientBoosting":
            self.model = GradientBoostingRegressor(**params["model_params"])
        elif params["model_name"] == "kNN":
            self.model = KNeighborsRegressor(**params["model_params"])
        elif params["model_name"] == "Ridge":
            self.model = Ridge(**params["model_params"])
        elif params["model_name"] == "Lasso":
            self.model = Lasso(**params["model_params"])
        elif params["model_name"] == "ExtraTrees":
            self.model = ExtraTreesRegressor(**params["model_params"])
        elif params["model_name"] == "AdaBoost":
            self.model = AdaBoostRegressor(**params["model_params"])
        else:
            self.model = None
            print(params)
            raise
        if self.debug:
            print(self.model)

    def _fit_and_predict_core(
        self, x, y=None, fitting=False, proba=False, support=None, score=False
    ):
        if support is None:
            if fitting == True:
                self.standardizer.fit(x)

            self.standardizer.transform(x)
            if score:
                pred = np.array(self.model.predict(x))
                return r2_score(pred.flatten(), np.array(y).flatten())

            if fitting == True:
                self.model.fit(x, y)
            if y is None:
                if proba:
                    return self.model.predict_proba(x)
                else:
                    return self.model.predict(x)
        else:
            if fitting == True:
                self.standardizer.fit(x.iloc[:, support])

            self.standardizer.transform(x.iloc[:, support])
            if score:
                pred = np.array(self.model.predict(x.iloc[:, support]))
                return r2_score(pred.flatten(), np.array(y).flatten())

            if fitting == True:
                self.model.fit(x.iloc[:, support], y)

            if y is None:
                if proba:
                    return self.model.predict_proba(x.iloc[:, support])
                else:
                    return self.model.predict(x.iloc[:, support])

        return None

    @on_timeout(limit=600, handler=handler_func, hint=u"regressor.fit")
    def fit(self, x, y, support=None):
        self._fit_and_predict_core(x, y, fitting=True, support=support)
        return self

    def predict(self, x, support=None):
        pred_y = self._fit_and_predict_core(x, support=support)
        return pred_y

    def predict_proba(self, x, support=None):
        pred_y = self._fit_and_predict_core(x, proba=True, support=support)
        return pred_y

    def score(self, x, y, support=None):
        return self._fit_and_predict_core(x, y, support=support, score=True)

class NullScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        return x 


