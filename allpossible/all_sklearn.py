import timeit
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.cross_decomposition import PLSRegression


from sklearn import metrics
class Objective:
    def __init__(self, X_train, X_test, y_train, y_test,
                 classifier_names = ['RandomForest', 'SVC', 'MLP', 'LogisticRegression', 'GradientBoosting'],
                 regressor_names = ['RandomForest', 'SVR', 'MLP', 'LinearRegression', 'PLS', 'GradientBoosting'],
                 classification_metrics = "f1_score"
                 ):
        self.x_train = X_train
        self.x_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.best_scores = {}
        self.best_params = {}
        self.best_models = {}
        self.best_score = 0
        self.best_model = None
        self.classifier_names = classifier_names
        self.regressor_names = regressor_names
        self.classification_metrics = classification_metrics
        self.times = {}
        self.scores = {}

    def __call__(self, trial):
        params = self.generate_params(trial, self.x_train)

        if len(set(self.y_train)) < len(self.y_train) / 10:
            model = Classifier(params)
            #model.fit(self.x_train, self.y_train)
            seconds = timeit.timeit(lambda: model.fit(self.x_train, self.y_train), number=1)
            if params['classifier_name'] not in self.times.keys():
                self.times[params['classifier_name']] = []
            self.times[params['classifier_name']].append(seconds)
            
            if self.classification_metrics == "f1_score":
                score = metrics.f1_score(self.y_test, model.predict(self.x_test))
            else:
                score = model.model.score(self.x_test, self.y_test)
            if params['classifier_name'] not in self.scores.keys():
                self.scores[params['classifier_name']] = []
            self.scores[params['classifier_name']].append(score)
            
            if self.best_score < score:
                self.best_score = score
                self.best_model = model
            if params['classifier_name'] not in self.best_scores.keys():
                self.best_scores[params['classifier_name']] = 0
            if self.best_scores[params['classifier_name']] < score:
                self.best_scores[params['classifier_name']] = score
                self.best_models[params['classifier_name']] = model
        else:
            model = Regressor(params)
            #model.fit(self.x_train, self.y_train)
            seconds = timeit.timeit(lambda: model.fit(self.x_train, self.y_train), number=1)
            if params['regressor_name'] not in self.times.keys():
                self.times[params['regressor_name']] = []
            self.times[params['regressor_name']].append(seconds)
            
            score = model.model.score(self.x_test, self.y_test)
            if params['regressor_name'] not in self.scores.keys():
                self.scores[params['regressor_name']] = []
            self.scores[params['regressor_name']].append(score)
            
            if self.best_score < score:
                self.best_score = score
                self.best_model = model
            if params['regressor_name'] not in self.best_scores.keys():
                self.best_scores[params['regressor_name']] = 0
            if self.best_scores[params['regressor_name']] < score:
                self.best_scores[params['regressor_name']] = score
                self.best_models[params['regressor_name']] = model
        return score


    def generate_params(self, trial, x):
        params = {}

        params['standardize'] = trial.suggest_categorical('standardize', ['NoScaler', 'StandardScaler', 'MinMaxScaler'])
        if len(set(self.y_train)) < len(self.y_train) / 10:
            params['classifier_name'] = trial.suggest_categorical('classifier_name', self.classifier_names)
            classifier_params = {}
            if params['classifier_name'] == 'SVC':
                classifier_params['kernel'] = trial.suggest_categorical('svc_kernel',
                                                                ['linear', 'rbf'])
                classifier_params['C'] = trial.suggest_loguniform('svc_c', 1e-10, 1e10)
                if classifier_params['kernel'] == 'rbf':
                    classifier_params['gamma'] = trial.suggest_categorical('svc_gamma',
                                                            ['auto', 'scale'])
                else:
                    classifier_params['gamma'] = 'auto'

            elif params['classifier_name'] == 'RandomForest':
                classifier_params['n_estimators'] = trial.suggest_categorical(
                    'rf_n_estimators', [5, 10, 20, 30, 50, 100])
                classifier_params['max_features'] = trial.suggest_categorical(
                    'rf_max_features', ['auto', 0.2, 0.4, 0.6, 0.8])
                classifier_params['max_depth'] = int(
                    trial.suggest_loguniform('rf_max_depth', 2, 32))
                classifier_params['n_jobs'] = -1
            elif params['classifier_name'] == 'MLP':
                layers = []
                n_layers = trial.suggest_int('n_layers', 1, 10)
                for i in range(n_layers):
                    layers.append(trial.suggest_int(str(i), 10, 100))
                classifier_params['hidden_layer_sizes'] = set(layers)
                learning_rate_init, = trial.suggest_loguniform('learning_rate_init', 0.001, 0.1),
                classifier_params['learning_rate_init'] = learning_rate_init
                classifier_params['max_iter'] = 2000
                classifier_params['early_stopping'] =True
            elif params['classifier_name'] == 'LogisticRegression':
                classifier_params['C'] = trial.suggest_loguniform('lr_C', 0.00001, 1000)
                classifier_params['solver'] = trial.suggest_categorical('lr_solver', ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])
                classifier_params['max_iter'] = 2000
            elif params['classifier_name'] == 'GradientBoosting':
                classifier_params['loss'] = trial.suggest_categorical('loss', ['deviance', 'exponential'])
                classifier_params['learning_rate'] = trial.suggest_loguniform('learning_rate_init', 0.001, 0.1)
                classifier_params['n_estimators'] = trial.suggest_categorical(
                    'gb_n_estimators', [5, 10, 20, 30, 50, 100])
                classifier_params['max_depth'] = int(
                    trial.suggest_loguniform('gb_max_depth', 2, 32))
            else:
                raise RuntimeError('unspport classifier', params['classifier_name'])
            params['classifier_params'] = classifier_params

        else:
            params['regressor_name'] = trial.suggest_categorical('regressor_name', self.regressor_names)
            regressor_params = {}
            if params['regressor_name'] == 'SVR':
                regressor_params['kernel'] = trial.suggest_categorical('svc_kernel',
                                                                ['linear', 'rbf'])
                regressor_params['C'] = trial.suggest_loguniform('svc_c', 1e-10, 1e10)
                if regressor_params['kernel'] == 'rbf':
                    regressor_params['gamma'] = trial.suggest_categorical('svc_gamma',
                                                            ['auto', 'scale'])
                else:
                    regressor_params['gamma'] = 'auto'

            elif params['regressor_name'] == 'RandomForest':
                regressor_params['n_estimators'] = trial.suggest_categorical(
                    'rf_n_estimators', [5, 10, 20, 30, 50, 100])
                regressor_params['max_features'] = trial.suggest_categorical(
                    'rf_max_features', ['auto', 0.2, 0.4, 0.6, 0.8])
                regressor_params['max_depth'] = int(
                    trial.suggest_loguniform('rf_max_depth', 2, 32))
                regressor_params['n_jobs'] = -1
            elif params['regressor_name'] == 'MLP':
                layers = []
                n_layers = trial.suggest_int('n_layers', 1, 10)
                for i in range(n_layers):
                    layers.append(trial.suggest_int(str(i), 10, 100))
                regressor_params['hidden_layer_sizes'] = set(layers)
                learning_rate_init, = trial.suggest_loguniform('learning_rate_init', 0.001, 0.1),
                regressor_params['learning_rate_init'] = learning_rate_init
                regressor_params['max_iter'] = 2000
                regressor_params['early_stopping'] =True
            elif params['regressor_name'] == 'PLS':
                regressor_params['n_components'] = trial.suggest_int("n_components", 2, self.x_train.shape[1])
                regressor_params['max_iter'] = 2000
            elif params['regressor_name'] == 'LinearRegression':
                pass
            elif params['regressor_name'] == 'GradientBoosting':
                #regressor_params['loss'] = trial.suggest_categorical('loss', ['deviance', 'exponential'])
                regressor_params['learning_rate'] = trial.suggest_loguniform('learning_rate_init', 0.001, 0.1)
                regressor_params['n_estimators'] = trial.suggest_categorical(
                    'gb_n_estimators', [5, 10, 20, 30, 50, 100])
                regressor_params['max_depth'] = int(
                    trial.suggest_loguniform('gb_max_depth', 2, 32))
            else:
                raise RuntimeError('unspport regressor', params['regressor_name'])
            params['regressor_params'] = regressor_params

        return params
        
#from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
class Classifier:
    def __init__(self, params):
        self.params = params
        if params['standardize'] == 'StandardScaler':
            self.standardizer = StandardScaler()
        elif params['standardize'] == 'MinMaxScaler':
            self.standardizer = MinMaxScaler()
        elif params['standardize'] == 'NoScaler':
            self.standardizer = NullScaler()

        if params['classifier_name'] == 'RandomForest':
            self.model = RandomForestClassifier(**params['classifier_params'])
        elif params['classifier_name'] == 'SVC':
            self.model = SVC(**params['classifier_params'])
        elif params['classifier_name'] == 'MLP':
            self.model = MLPClassifier(**params['classifier_params'])
        elif params['classifier_name'] == 'LogisticRegression':
            self.model = LogisticRegression(**params['classifier_params'])
        elif params['classifier_name'] == 'GradientBoosting':
            self.model = GradientBoostingClassifier(**params['classifier_params'])

    def _fit_and_predict_core(self, x, y=None, fitting=False, proba=False):
        if fitting == True:
            self.standardizer.fit(x)
        self.standardizer.transform(x)

        if fitting == True:
            self.model.fit(x, y)
        if y is None:
            if proba:
                return self.model.predict_proba(x)
            else:
                return self.model.predict(x)
        return None

    def fit(self, x, y):
        self._fit_and_predict_core(x, y, fitting=True)
        return self

    def predict(self, x):
        pred_y = self._fit_and_predict_core(x)
        return pred_y

    def predict_proba(self, x):
        pred_y = self._fit_and_predict_core(x, proba=True)
        return pred_y
        
#from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
class Regressor:
    def __init__(self, params):
        self.params = params
        if params['standardize'] == 'StandardScaler':
            self.standardizer = StandardScaler()
        elif params['standardize'] == 'MinMaxScaler':
            self.standardizer = MinMaxScaler()
        elif params['standardize'] == 'NoScaler':
            self.standardizer = NullScaler()

        if params['regressor_name'] == 'RandomForest':
            self.model = RandomForestRegressor(**params['regressor_params'])
        elif params['regressor_name'] == 'SVR':
            self.model = SVR(**params['regressor_params'])
        elif params['regressor_name'] == 'MLP':
            self.model = MLPRegressor(**params['regressor_params'])
        elif params['regressor_name'] == 'LinearRegression':
            self.model = LinearRegression(**params['regressor_params'])
        elif params['regressor_name'] == 'PLS':
            self.model = PLSRegression(**params['regressor_params'])
        elif params['regressor_name'] == 'GradientBoosting':
            self.model = GradientBoostingRegressor(**params['regressor_params'])

    def _fit_and_predict_core(self, x, y=None, fitting=False, proba=False):
        if fitting == True:
            self.standardizer.fit(x)
        self.standardizer.transform(x)

        if fitting == True:
            self.model.fit(x, y)
        if y is None:
            if proba:
                return self.model.predict_proba(x)
            else:
                return self.model.predict(x)
        return None

    def fit(self, x, y):
        self._fit_and_predict_core(x, y, fitting=True)
        return self

    def predict(self, x):
        pred_y = self._fit_and_predict_core(x)
        return pred_y

    def predict_proba(self, x):
        pred_y = self._fit_and_predict_core(x, proba=True)
        return pred_y


class NullScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        return x

def objective_summary(objective):
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16, 8))

    names = [n for n in reversed(sorted(list(objective.scores.keys())))]

    score_means = []
    score_stds = []
    second_means = []
    second_stds = []
    selected = []
    sum_second = []
    for name in names:
        score_means.append(np.array(objective.scores[name]).mean())
        score_stds.append(np.array(objective.scores[name]).std())
        second_means.append(np.array(objective.times[name]).mean())
        second_stds.append(np.array(objective.times[name]).std())
        selected.append(len(objective.times[name]))
        sum_second.append(sum(objective.times[name]))

    axes[0].barh(names, score_means, xerr=score_stds)
    axes[0].set_xlabel('score')
    axes[0].grid()
    axes[1].barh(names, selected)
    axes[1].set_xlabel('selected (times)')
    axes[1].grid()
    axes[1].yaxis.set_visible(False)
    axes[2].barh(names, second_means, xerr=second_stds)
    axes[2].set_xlabel('calculation time (seconds)')
    axes[2].grid()
    axes[2].yaxis.set_visible(False)
    axes[3].barh(names, sum_second)
    axes[3].set_xlabel('sum calculation time (seconds)')
    axes[3].grid()
    axes[3].yaxis.set_visible(False)
    plt.show()
