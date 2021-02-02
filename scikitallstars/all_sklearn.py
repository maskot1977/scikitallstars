import time
import timeit
from scikitallstars.timeout import on_timeout
import scikitallstars.timeout_decorator as timeout_decorator
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
from sklearn.model_selection import train_test_split

def handler_func(msg):
        print(msg)

class Objective:
    def __init__(self, 
                 x_train, 
                 y_train,
                 x_test = None, 
                 y_test = None,
                 classifier_names = ['RandomForest', 'SVC', 'MLP', 'GradientBoosting', 'LogisticRegression'],
                 regressor_names = ['RandomForest', 'SVR', 'MLP', 'GradientBoosting', 'LinearRegression', 'PLS'],
                 classification_metrics = "f1_score"
                 ):
        self.x_train = x_train
        self.x_test = x_test
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
        self.debug = False
        
        self.gb_loss = ['deviance', 'exponential']
        self.gb_learning_rate_init = [0.001, 0.1]
        self.gb_n_estimators = [100]
        self.gb_max_depth = [2, 32]
        
        self.lr_C = [0.00001, 1000]
        self.lr_max_iter = 530000
        self.lr_solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
        
        self.mlp_max_iter = 530000
        self.mlp_n_layers = [1, 10]
        self.mlp_n_neurons = [10, 100]
        
        self.pls_max_iter = 530000

        self.rf_max_depth = [2, 32]
        self.rf_max_features = ['auto']
        self.rf_n_estimators = [100]

        self.svm_kernel = ['linear', 'rbf']
        self.svm_c = [1e-5, 1e5]
        self.svm_max_iter = 53000000


    #@on_timeout(limit=5, handler=handler_func, hint=u'call')
    @timeout_decorator.timeout(5)
    def __call__(self, trial):
        if self.y_test is None:
                x_train, x_test, y_train, y_test = train_test_split(self.x_train, self.y_train, test_size=0.2)
        else:
                x_train = self.x_train
                x_test = self.x_test
                y_train = self.y_train
                y_test = self.y_test
                
        params = self.generate_params(trial, x_train)

        if len(set(y_train)) < len(y_train) / 10:
            model = Classifier(params, debug=self.debug)
            seconds = self.model_fit(model, x_train, y_train)
            if params['classifier_name'] not in self.times.keys():
                self.times[params['classifier_name']] = []
            self.times[params['classifier_name']].append(seconds)
            
            if self.classification_metrics == "f1_score":
                score = metrics.f1_score(y_test, model.predict(x_test))
            else:
                score = model.model.score(x_test, y_test)
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
            model = Regressor(params, debug=self.debug)
            seconds = self.model_fit(model, x_train, y_train)
            if params['regressor_name'] not in self.times.keys():
                self.times[params['regressor_name']] = []
            self.times[params['regressor_name']].append(seconds)
            
            score = model.model.score(x_test, y_test)
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


    @on_timeout(limit=5, handler=handler_func, hint=u'model_fit')
    def model_fit(self, model, x_train, y_train):
        return timeit.timeit(lambda: model.fit(x_train, y_train), number=1)
    
    def generate_params(self, trial, x):
        params = {}

        params['standardize'] = trial.suggest_categorical('standardize', ['NoScaler', 'StandardScaler', 'MinMaxScaler'])
        if len(set(self.y_train)) < len(self.y_train) / 10:
            params['classifier_name'] = trial.suggest_categorical('classifier_name', self.classifier_names)
            classifier_params = {}
                
            if params['classifier_name'] == 'SVC':
                classifier_params['kernel'] = trial.suggest_categorical(
                        'svc_kernel', ['linear', 'rbf'])
                classifier_params['C'] = trial.suggest_loguniform(
                        'svm_c', self.svm_c[0], self.svm_c[1])
                if classifier_params['kernel'] == 'rbf':
                    classifier_params['gamma'] = trial.suggest_categorical(
                            'svc_gamma',['auto', 'scale'])
                else:
                    classifier_params['gamma'] = 'auto'
                classifier_params['max_iter'] = self.svm_max_iter

            elif params['classifier_name'] == 'RandomForest':
                classifier_params['n_estimators'] = trial.suggest_categorical(
                    'rf_n_estimators', self.rf_n_estimators)
                classifier_params['max_features'] = trial.suggest_categorical(
                    'rf_max_features', self.rf_max_features)
                classifier_params['n_jobs'] = -1
                classifier_params['max_depth'] = int(
                    trial.suggest_int('rf_max_depth', self.rf_max_depth[0], self.rf_max_depth[1]))
                
            elif params['classifier_name'] == 'MLP':
                layers = []
                n_layers = trial.suggest_int(
                        'n_layers', self.mlp_n_layers[0], self.mlp_n_layers[1])
                for i in range(n_layers):
                    layers.append(trial.suggest_int(
                            str(i), self.mlp_n_neurons[0], self.mlp_n_neurons[1]))
                classifier_params['hidden_layer_sizes'] = set(layers)
                classifier_params['max_iter'] = self.mlp_max_iter
                classifier_params['early_stopping'] = True
                
            elif params['classifier_name'] == 'LogisticRegression':
                classifier_params['C'] = trial.suggest_loguniform(
                        'lr_C', self.lr_C[0], self.lr_C[0])
                classifier_params['solver'] = trial.suggest_categorical(
                        'lr_solver', self.lr_solver)
                classifier_params['max_iter'] = self.lr_max_iter
                
            elif params['classifier_name'] == 'GradientBoosting':
                classifier_params['loss'] = trial.suggest_categorical('loss', self.gb_loss)
                classifier_params['learning_rate'] = trial.suggest_loguniform(
                        'learning_rate_init', self.learning_rate_init[0], self.learning_rate_init[1])
                classifier_params['n_estimators'] = trial.suggest_categorical(
                    'gb_n_estimators', self.gb_n_estimators)
                classifier_params['max_depth'] = int(
                    trial.suggest_int('gb_max_depth', self.gb_max_depth[0], self.gb_max_depth[1]))
                
            else:
                raise RuntimeError('unspport classifier', params['classifier_name'])
            params['classifier_params'] = classifier_params

        else:
            params['regressor_name'] = trial.suggest_categorical('regressor_name', self.regressor_names)
            #print(params['regressor_name'])
            regressor_params = {}
            if params['regressor_name'] == 'SVR':
                regressor_params['kernel'] = trial.suggest_categorical(
                        'svm_kernel', self.svm_kernel)
                regressor_params['C'] = trial.suggest_loguniform(
                        'svm_c', self.svm_c[0], self.svm_c[1])
                if regressor_params['kernel'] == 'rbf':
                    regressor_params['gamma'] = trial.suggest_categorical('svc_gamma',
                                                            ['auto', 'scale'])
                else:
                    regressor_params['gamma'] = 'auto'
                regressor_params['max_iter'] = self.svm_max_iter

            elif params['regressor_name'] == 'RandomForest':
                regressor_params['n_estimators'] = trial.suggest_categorical(
                    'rf_n_estimators', self.rf_n_estimators)
                regressor_params['max_features'] = trial.suggest_categorical(
                    'rf_max_features', self.rf_max_features)
                regressor_params['max_depth'] = trial.suggest_int(
                        'rf_max_depth', self.rf_max_depth[0], self.rf_max_depth[1])
                #regressor_params['n_jobs'] = -1
                
            elif params['regressor_name'] == 'MLP':
                layers = []
                n_layers = trial.suggest_int(
                        'n_layers', self.mlp_n_layers[0], self.mlp_n_layers[1])
                for i in range(n_layers):
                    layers.append(trial.suggest_int(
                            str(i), self.mlp_n_neurons[0], self.mlp_n_neurons[1]))
                regressor_params['hidden_layer_sizes'] = set(layers)
                regressor_params['max_iter'] = self.mlp_max_iter
                regressor_params['early_stopping'] =True
                
            elif params['regressor_name'] == 'PLS':
                regressor_params['n_components'] = trial.suggest_int("n_components", 2, self.x_train.shape[1])
                regressor_params['max_iter'] = self.pls_max_iter
                
            elif params['regressor_name'] == 'LinearRegression':
                pass
        
            elif params['regressor_name'] == 'GradientBoosting':
                regressor_params['learning_rate'] = trial.suggest_loguniform(
                        'learning_rate_init', self.gb_learning_rate_init[0], self.gb_learning_rate_init[1])
                regressor_params['n_estimators'] = trial.suggest_categorical(
                    'gb_n_estimators', self.gb_n_estimators)
                regressor_params['max_depth'] = int(
                    trial.suggest_loguniform('gb_max_depth', self.gb_max_depth[0], self.gb_max_depth[1]))
            else:
                raise RuntimeError('unspport regressor', params['regressor_name'])
            params['regressor_params'] = regressor_params

        return params
        
#from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
class Classifier:
    def __init__(self, params, debug=False):
        self.params = params
        self.debug = debug
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
        if self.debug:
            print(self.model)
        
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

    @on_timeout(limit=60, handler=handler_func, hint=u'classifier.fit')
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
    def __init__(self, params, debug=False):
        self.params = params
        self.debug = debug
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
        if self.debug:
            print(self.model)

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

    @on_timeout(limit=60, handler=handler_func, hint=u'regressor.fit')
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
    axes[0].set_xlim([0.0, 1.0])
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
    axes[3].set_xlabel('total calculation time (seconds)')
    axes[3].grid()
    axes[3].yaxis.set_visible(False)
    plt.show()


