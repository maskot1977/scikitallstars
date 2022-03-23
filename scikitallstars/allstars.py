import time
import timeit

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn import metrics


from sklearn.feature_selection import SelectFromModel



import scikitallstars.timeout_decorator as timeout_decorator
from scikitallstars.estimators import Classifier, Regressor
from scikitallstars.timeout import on_timeout, handler_func
from sklearn.model_selection import train_test_split





class Objective:
    def __init__(
        self,
        x_train,
        y_train,
        x_valid=None,
        y_valid=None,
        support=None,
        classifier_names=[
            "LogisticRegression",
            "LDA",
            "QDA",
            "Ridge",
            "SVC",
            "MLP",
            "kNN",
            "AdaBoost",
            "RandomForest",
            "ExtraTrees",
            "GradientBoosting",
        ],
        regressor_names=[
            "LinearRegression",
            "PLS",
            "Lasso",
            "Ridge",
            "SVR",
            "MLP",
            "kNN",
            "AdaBoost",
            "RandomForest",
            "ExtraTrees",
            "GradientBoosting",
        ],
        classification_metrics="f1_score",
        test_size=0.1,
        split_random_state=None
    ):
        self.x_train = x_train
        self.x_valid = x_valid
        self.y_train = y_train
        self.y_valid = y_valid
        self.support = support
        self.best_scores = {}
        self.best_params = {}
        self.best_models = {}
        self.best_score = 0
        self.best_model = None
        self.test_size = test_size
        self.split_random_state = split_random_state
        self.classifier_names = classifier_names
        self.regressor_names = regressor_names
        self.classification_metrics = classification_metrics
        self.times = {}
        self.scores = {}
        self.debug = False
        self.scalers = ["StandardScaler", "MinMaxScaler"]
        self.is_regressor = True
        if len(set(y_train)) < 3:
            self.is_regressor = False

        self.gb_loss = ["deviance", "exponential"]
        self.gb_learning_rate_init = [0.001, 0.1]
        self.gb_n_estimators = [50, 200]
        self.gb_max_depth = [2, 32]
        self.gb_warm_start = [True, False]

        self.et_n_estimators = [50, 300]
        self.et_max_depth = [2, 32]
        self.et_warm_start = [True, False]

        self.ab_n_estimators = [50, 300]
        self.ab_loss = ["linear", "square", "exponential"]

        self.knn_n_neighbors = [2, 10]
        self.knn_weights = ["uniform", "distance"]
        self.knn_algorithm = ["auto"] #, "ball_tree", "kd_tree", "brute"]
        self.knn_leaf_size = [20, 40]

        self.lr_C = [1e-5, 1e5]
        self.lr_max_iter = 530000
        self.lr_solver = ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]

        self.mlp_max_iter = 530000
        self.mlp_n_layers = [1, 10]
        self.mlp_n_neurons = [4, 64]
        self.mlp_warm_start = [True, False]
        self.mlp_activation = ["relu"] #, "identity", "logistic", "tanh"]

        self.pls_max_iter = 530000
        self.pls_scale = [True, False]
        self.pls_algorithm = ["nipals", "svd"]
        self.pls_tol = [1e-7, 1e-5]

        self.lasso_alpha = [1e-5, 1e5]
        self.lasso_max_iter = 530000
        self.lasso_warm_start = [True, False]
        self.lasso_normalize = [True, False]
        self.lasso_selection = ["cyclic", "random"]

        self.ridge_alpha = [1e-5, 1e5]
        self.ridge_max_iter = 530000
        self.ridge_solver = ["auto"]
        #    "auto",
        #    "svd",
        #    "cholesky",
        #    "lsqr",
        #    "sparse_cg",
        #    "sag",
        #    "saga",
        #]
        self.ridge_normalize = [True, False]

        self.rf_max_depth = [2, 32]
        self.rf_max_features = ["auto"] #, "sqrt", "log2"]
        self.rf_n_estimators = [100, 200]
        self.rf_warm_start = [True, False]

        self.svm_kernel = ["linear", "rbf"]
        self.svm_c = [1e-5, 1e5]
        self.svm_epsilon = [1e-5, 1e5]
        self.svm_max_iter = 530000

        self.linear_regression_fit_intercept = [True, False]
        self.linear_regression_normalize = [True, False]

    def get_model_names(self):
        if self.is_regressor:
            return self.regressor_names
        else:
            return self.classifier_names

    def set_model_names(self, model_names):
        if self.is_regressor:
            self.regressor_names = model_names
        else:
            self.classifier_names = model_names

    # @on_timeout(limit=5, handler=handler_func, hint=u'call')
    @timeout_decorator.timeout(10)
    def __call__(self, trial):
        if self.support is None:
            if self.y_valid is None:
                x_train, x_valid, y_train, y_valid = train_test_split(
                    self.x_train, self.y_train, test_size=self.test_size
                )
            else:
                x_train = self.x_train
                x_valid = self.x_valid
                y_train = self.y_train
                y_valid = self.y_valid
        else:
            if self.y_valid is None:
                x_train, x_valid, y_train, y_valid = train_test_split(
                    self.x_train.iloc[:, self.support], self.y_train, test_size=self.test_size, random_state=self.split_random_state
                )
            else:
                x_train = self.x_train.iloc[:, self.support]
                x_valid = self.x_valid.iloc[:, self.support]
                y_train = self.y_train
                y_valid = self.y_valid

        params = self.generate_params(trial, x_train)

        if len(set(y_train)) < 3:
            self.is_regressor = False
            model = Classifier(params, debug=self.debug)
            seconds = self.model_fit(model, x_train, y_train)
            if params["model_name"] not in self.times.keys():
                self.times[params["model_name"]] = []
            self.times[params["model_name"]].append(seconds)

            if self.classification_metrics == "f1_score":
                if self.support is None:
                    score = metrics.f1_score(model.predict(x_valid), y_valid)
                    # score = metrics.f1_score(model.predict(self.x_train), self.y_train)
                else:
                    # score = metrics.f1_score(model.predict(self.x_train.iloc[:, self.support]), self.y_train)
                    # score = metrics.f1_score(model.predict(x_valid.iloc[:, self.support]), y_valid)
                    score = metrics.f1_score(model.predict(x_valid), y_valid)
            else:
                if self.support is None:
                    score = model.model.score(x_valid, y_valid)
                    # score = model.model.score(self.x_train, self.y_train)
                else:
                    # score = model.model.score(self.x_train.iloc[:, self.support], self.y_train)
                    score = model.model.score(x_valid, y_valid)

            if params["model_name"] not in self.scores.keys():
                self.scores[params["model_name"]] = []
            self.scores[params["model_name"]].append(score)

            if self.best_score < score:
                self.best_score = score
                self.best_model = model
            if params["model_name"] not in self.best_scores.keys():
                self.best_scores[params["model_name"]] = 0
            if self.best_scores[params["model_name"]] < score:
                self.best_scores[params["model_name"]] = score
                self.best_models[params["model_name"]] = model

        else:
            self.is_regressor = True
            model = Regressor(params, debug=self.debug, support=self.support)
            seconds = self.model_fit(model, x_train, y_train)
            if params["model_name"] not in self.times.keys():
                self.times[params["model_name"]] = []
            self.times[params["model_name"]].append(seconds)

            if self.support is None:
                # score = model.model.score(self.x_train, self.y_train)
                score = model.model.score(x_valid, y_valid)
            else:
                # score = model.model.score(self.x_train.iloc[:, self.support], self.y_train)
                score = model.model.score(x_valid, y_valid)
            if params["model_name"] not in self.scores.keys():
                self.scores[params["model_name"]] = []
            self.scores[params["model_name"]].append(score)

            if self.best_score < score:
                self.best_score = score
                self.best_model = model
            if params["model_name"] not in self.best_scores.keys():
                self.best_scores[params["model_name"]] = 0
            if self.best_scores[params["model_name"]] < score:
                self.best_scores[params["model_name"]] = score
                self.best_models[params["model_name"]] = model

        return score

    @on_timeout(limit=600, handler=handler_func, hint=u"model_fit")
    def model_fit(self, model, x_train, y_train):
        return timeit.timeit(lambda: model.fit(x_train, y_train), number=1)

    def generate_params(self, trial, x):
        params = {}

        params["standardize"] = trial.suggest_categorical("standardize", self.scalers)
        if len(set(self.y_train)) < 3:
            params["model_name"] = trial.suggest_categorical(
                "model_name", self.classifier_names
            )
            model_params = {}

            if params["model_name"] == "SVC":
                model_params["kernel"] = trial.suggest_categorical(
                    "svc_kernel", ["linear", "rbf"]
                )
                model_params["C"] = trial.suggest_loguniform(
                    "svm_c", self.svm_c[0], self.svm_c[1]
                )
                # model_params["epsilon"] = trial.suggest_loguniform(
                #    "svm_epsilon", self.svm_epsilon[0], self.svm_epsilon[1]
                # )
                if model_params["kernel"] == "rbf":
                    model_params["gamma"] = trial.suggest_categorical(
                        "svc_gamma", ["auto", "scale"]
                    )
                else:
                    model_params["gamma"] = "auto"
                model_params["max_iter"] = self.svm_max_iter
                model_params["probability"] = True

            elif params["model_name"] == "RandomForest":
                model_params["n_estimators"] = trial.suggest_int(
                    "rf_n_estimators", self.rf_n_estimators[0], self.rf_n_estimators[1]
                )
                model_params["max_features"] = trial.suggest_categorical(
                    "rf_max_features", self.rf_max_features
                )
                model_params["n_jobs"] = -1
                model_params["max_depth"] = trial.suggest_int(
                    "rf_max_depth", self.rf_max_depth[0], self.rf_max_depth[1]
                )
                model_params["warm_start"] = trial.suggest_categorical(
                    "rf_warm_start", self.rf_warm_start
                )

            elif params["model_name"] == "MLP":
                layers = []
                n_layers = trial.suggest_int(
                    "n_layers", self.mlp_n_layers[0], self.mlp_n_layers[1]
                )
                for i in range(n_layers):
                    layers.append(
                        trial.suggest_int(
                            str(i), self.mlp_n_neurons[0], self.mlp_n_neurons[1]
                        )
                    )
                model_params["hidden_layer_sizes"] = set(layers)
                model_params["max_iter"] = self.mlp_max_iter
                model_params["early_stopping"] = True
                model_params["warm_start"] = trial.suggest_categorical(
                    "mlp_warm_start", self.mlp_warm_start
                )
                model_params["activation"] = trial.suggest_categorical(
                    "mlp_activation", self.mlp_activation
                )

            elif params["model_name"] == "LogisticRegression":
                model_params["C"] = trial.suggest_loguniform(
                    "lr_C", self.lr_C[0], self.lr_C[0]
                )
                model_params["solver"] = trial.suggest_categorical(
                    "lr_solver", self.lr_solver
                )
                model_params["max_iter"] = self.lr_max_iter

            elif params["model_name"] == "GradientBoosting":
                model_params["loss"] = trial.suggest_categorical("loss", self.gb_loss)
                model_params["n_estimators"] = trial.suggest_int(
                    "gb_n_estimators", self.gb_n_estimators[0], self.gb_n_estimators[1]
                )
                model_params["max_depth"] = trial.suggest_int(
                    "gb_max_depth", self.gb_max_depth[0], self.gb_max_depth[1]
                )
                model_params["warm_start"] = trial.suggest_categorical(
                    "gb_warm_start", self.gb_warm_start
                )

            elif params["model_name"] == "ExtraTrees":
                model_params["n_estimators"] = trial.suggest_int(
                    "et_n_estimators", self.et_n_estimators[0], self.et_n_estimators[1]
                )
                model_params["max_depth"] = trial.suggest_int(
                    "et_max_depth", self.et_max_depth[0], self.et_max_depth[1]
                )
                model_params["warm_start"] = trial.suggest_categorical(
                    "et_warm_start", self.et_warm_start
                )

            elif params["model_name"] == "AdaBoost":
                model_params["n_estimators"] = trial.suggest_int(
                    "ab_n_estimators", self.ab_n_estimators[0], self.ab_n_estimators[1]
                )
                # model_params["loss"] = trial.suggest_categorical(
                #    "ab_loss", self.ab_loss
                # )

            elif params["model_name"] == "kNN":
                model_params["n_neighbors"] = trial.suggest_int(
                    "knn_n_neighbors", self.knn_n_neighbors[0], self.knn_n_neighbors[1]
                )
                model_params["weights"] = trial.suggest_categorical(
                    "knn_weights", self.knn_weights
                )
                model_params["algorithm"] = trial.suggest_categorical(
                    "knn_algorithm", self.knn_algorithm
                )
                model_params["leaf_size"] = trial.suggest_int(
                    "knn_leaf_size", self.knn_leaf_size[0], self.knn_leaf_size[1]
                )

            elif params["model_name"] == "Ridge":
                model_params["alpha"] = trial.suggest_loguniform(
                    "ridge_alpha", self.ridge_alpha[0], self.ridge_alpha[1]
                )
                model_params["max_iter"] = self.ridge_max_iter
                model_params["normalize"] = trial.suggest_categorical(
                    "ridge_normalize", self.ridge_normalize
                )
                model_params["solver"] = trial.suggest_categorical(
                    "ridge_solver", self.ridge_solver
                )

            elif params["model_name"] == "QDA":
                pass
            elif params["model_name"] == "LDA":
                pass
            else:
                raise RuntimeError("unspport classifier", params["model_name"])
            params["model_params"] = model_params

        else:
            params["model_name"] = trial.suggest_categorical(
                "model_name", self.regressor_names
            )
            model_params = {}

            if params["model_name"] == "GradientBoosting":
                # model_params["loss"] = trial.suggest_categorical(
                #    "gb_loss", ["ls", "lad", "huber", "quantile"]
                # )
                model_params["learning_rate"] = trial.suggest_loguniform(
                    "learning_rate_init",
                    self.gb_learning_rate_init[0],
                    self.gb_learning_rate_init[1],
                )
                model_params["n_estimators"] = trial.suggest_int(
                    "gb_n_estimators", self.gb_n_estimators[0], self.gb_n_estimators[1]
                )
                # model_params["criterion"] = trial.suggest_categorical(
                #    "gb_criterion", ["friedman_mse", "mse", "mae"]
                # )
                model_params["max_depth"] = trial.suggest_int(
                    "gb_max_depth", self.gb_max_depth[0], self.gb_max_depth[1]
                )
                model_params["warm_start"] = trial.suggest_categorical(
                    "gb_warm_start", self.gb_warm_start
                )
                # model_params["max_features"] = trial.suggest_categorical(
                #    "gb_max_features", ["auto", "sqrt", "log2"]
                # )
                # model_params["tol"] = trial.suggest_loguniform(
                #    "gb_tol", 1e-5, 1e-3
                # )

            elif params["model_name"] == "ExtraTrees":
                model_params["n_estimators"] = trial.suggest_int(
                    "et_n_estimators", self.et_n_estimators[0], self.et_n_estimators[1]
                )
                # model_params["criterion"] = trial.suggest_categorical(
                #    "et_criterion", ["mse", "mae"]
                # )
                model_params["max_depth"] = trial.suggest_int(
                    "et_max_depth", self.et_max_depth[0], self.et_max_depth[1]
                )
                model_params["max_features"] = trial.suggest_categorical(
                    "et_max_features", ["auto"] #, "sqrt", "log2"]
                )
                model_params["bootstrap"] = True
                model_params["oob_score"] = trial.suggest_categorical(
                    "et_oob_score", [True]
                )
                model_params["warm_start"] = trial.suggest_categorical(
                    "et_warm_start", self.et_warm_start
                )

            elif params["model_name"] == "RandomForest":
                model_params["n_estimators"] = trial.suggest_int(
                    "rf_n_estimators", self.rf_n_estimators[0], self.rf_n_estimators[1]
                )
                # model_params["criterion"] = trial.suggest_categorical(
                #    "rf_criterion", ["mse", "mae"]
                # )
                model_params["max_depth"] = trial.suggest_int(
                    "rf_max_depth", self.rf_max_depth[0], self.rf_max_depth[1]
                )
                model_params["max_features"] = trial.suggest_categorical(
                    "rf_max_features", self.rf_max_features
                )
                model_params["bootstrap"] = True
                model_params["oob_score"] = trial.suggest_categorical(
                    "rf_oob_score", [True]
                )
                model_params["warm_start"] = trial.suggest_categorical(
                    "rf_warm_start", self.rf_warm_start
                )

            elif params["model_name"] == "AdaBoost":
                model_params["n_estimators"] = trial.suggest_int(
                    "ab_n_estimators", self.ab_n_estimators[0], self.ab_n_estimators[1]
                )
                model_params["learning_rate"] = trial.suggest_loguniform(
                    "ab_learning_rate", 0.1, 1.0
                )
                model_params["loss"] = trial.suggest_categorical(
                    "ab_loss", self.ab_loss
                )

            elif params["model_name"] == "MLP":
                layers = []
                n_layers = trial.suggest_int(
                    "n_layers", self.mlp_n_layers[0], self.mlp_n_layers[1]
                )
                for i in range(n_layers):
                    layers.append(
                        trial.suggest_int(
                            str(i), self.mlp_n_neurons[0], self.mlp_n_neurons[1]
                        )
                    )
                model_params["hidden_layer_sizes"] = set(layers)
                # model_params["activation"] = trial.suggest_categorical(
                #    "mlp_activation", self.mlp_activation
                # )
                # model_params["solver"] = trial.suggest_categorical(
                #    "mlp_solver", ["sgd", "adam"]
                # )
                model_params["solver"] = "adam"
                model_params["learning_rate"] = trial.suggest_categorical(
                    "mlp_learning_rate", ["constant", "invscaling", "adaptive"]
                )
                if model_params["solver"] in ["sgd", "adam"]:
                    model_params["learning_rate_init"] = trial.suggest_loguniform(
                        "mlp_learning_rate_init", 1e-4, 1e-2
                    )
                model_params["max_iter"] = self.mlp_max_iter
                model_params["early_stopping"] = True
                model_params["warm_start"] = trial.suggest_categorical(
                    "mlp_warm_start", self.mlp_warm_start
                )

            elif params["model_name"] == "SVR":
                model_params["kernel"] = trial.suggest_categorical(
                    "svm_kernel", self.svm_kernel
                )
                model_params["C"] = trial.suggest_loguniform(
                    "svm_c", self.svm_c[0], self.svm_c[1]
                )
                if model_params["kernel"] == "rbf":
                    model_params["gamma"] = trial.suggest_categorical(
                        "svc_gamma", ["auto"] #, "scale"]
                    )
                else:
                    model_params["gamma"] = "auto"
                model_params["max_iter"] = self.svm_max_iter
                # model_params["epsilon"] = trial.suggest_loguniform(
                #    "svm_epsilon", self.svm_epsilon[0], self.svm_epsilon[1]
                # )

            elif params["model_name"] == "kNN":
                model_params["n_neighbors"] = trial.suggest_int(
                    "knn_n_neighbors", self.knn_n_neighbors[0], self.knn_n_neighbors[1]
                )
                model_params["weights"] = trial.suggest_categorical(
                    "knn_weights", self.knn_weights
                )
                model_params["algorithm"] = trial.suggest_categorical(
                    "knn_algorithm", self.knn_algorithm
                )

            elif params["model_name"] == "Ridge":
                model_params["alpha"] = trial.suggest_loguniform(
                    "ridge_alpha", self.ridge_alpha[0], self.ridge_alpha[1]
                )
                model_params["max_iter"] = self.ridge_max_iter
                # model_params["normalize"] = trial.suggest_categorical(
                #    "ridge_normalize", self.ridge_normalize
                #)
                model_params["solver"] = trial.suggest_categorical(
                    "ridge_solver", self.ridge_solver
                )

            elif params["model_name"] == "Lasso":
                model_params["alpha"] = trial.suggest_loguniform(
                    "lasso_alpha", self.lasso_alpha[0], self.lasso_alpha[1]
                )
                model_params["max_iter"] = self.lasso_max_iter
                model_params["warm_start"] = trial.suggest_categorical(
                    "lasso_warm_start", self.lasso_warm_start
                )
                # model_params["normalize"] = trial.suggest_categorical(
                #    "lasso_normalize", self.lasso_normalize
                #)
                model_params["selection"] = trial.suggest_categorical(
                    "lasso_selection", self.lasso_selection
                )

            elif params["model_name"] == "PLS":
                if self.support is None:
                    model_params["n_components"] = trial.suggest_int(
                        "n_components", 2, self.x_train.shape[1]
                    )
                else:
                    model_params["n_components"] = trial.suggest_int(
                        "n_components", 2, self.x_train.iloc[:, self.support].shape[1]
                    )
                model_params["max_iter"] = self.pls_max_iter
                model_params["scale"] = trial.suggest_categorical(
                    "pls_scale", self.pls_scale
                )
                # model_params["algorithm"] = trial.suggest_categorical(
                #    "pls_algorithm", self.pls_algorithm
                # )
                model_params["tol"] = trial.suggest_loguniform(
                    "pls_tol",
                    self.pls_tol[0],
                    self.pls_tol[1],
                )

            elif params["model_name"] == "LinearRegression":
                model_params["fit_intercept"] = trial.suggest_categorical(
                    "linear_regression_fit_intercept",
                    self.linear_regression_fit_intercept,
                )
                # model_params["normalize"] = trial.suggest_categorical(
                #    "linear_regression_normalize", self.linear_regression_normalize
                #)

            else:
                raise RuntimeError("unspport regressor", params["model_name"])
            params["model_params"] = model_params

        return params

    def predict(self, x):
        return self.best_model.predict(pd.DataFrame(x), support=self.support)

    def score(self, x, y):
        if type(y) is not pd.core.series.Series:
            try:
                y = pd.DataFrame(y)[0]
            except:
                pass
        return self.best_model.score(pd.DataFrame(x), y, support=self.support)









def fit(
    X_train,
    y_train,
    x_valid=None,
    y_valid=None,
    feature_selection=True,
    verbose=True,
    timeout=100,
    n_trials=100,
    show_progress_bar=True,
):
    X_train = pd.DataFrame(X_train)
    if type(y_train) is not pd.core.series.Series:
        y_train = pd.DataFrame(y_train)[0]
    if feature_selection:
        support = random_forest_feature_selector(X_train, y_train, x_valid=x_valid, y_valid=y_valid)
        X_train_selected = X_train.iloc[:, support]
        if verbose:
            print(
                "feature selection: X_train",
                X_train.shape,
                "->",
                X_train_selected.shape,
            )
        # X_train = X_train_selected
    else:
        support = np.array([True] * X_train.shape[1])
        if verbose:
            print("X_train", X_train.shape)

    objective = Objective(X_train, y_train, x_valid=x_valid, y_valid=y_valid, support=support)
    optuna.logging.set_verbosity(optuna.logging.WARN)
    study = optuna.create_study(direction="maximize")

    model_names = objective.get_model_names()
    for model_name in model_names:
        if verbose:
            print(model_name)
        for _ in range(n_trials):
            study.enqueue_trial({"model_name": model_name})

        study.optimize(
            objective,
            timeout=timeout,
            n_trials=n_trials,
            show_progress_bar=show_progress_bar,
        )
        if verbose:
            if model_name in objective.best_scores.keys():
                if model_name in objective.best_models.keys():
                    print(
                        objective.best_scores[model_name],
                        objective.best_models[model_name].model,
                    )

    study.optimize(
        objective,
        timeout=timeout,
        n_trials=n_trials,
        show_progress_bar=show_progress_bar,
    )

    if verbose:
        print(objective.best_scores)

    return objective


def random_forest_feature_selector(
    X_train, y_train, x_valid=None, y_valid=None, timeout=50, n_trials=100, show_progress_bar=False
):
    objective = Objective(X_train, y_train, x_valid=x_valid, y_valid=y_valid)
    objective.set_model_names(["RandomForest"])

    optuna.logging.set_verbosity(optuna.logging.WARN)
    study = optuna.create_study(direction="maximize")
    study.optimize(
        objective,
        timeout=timeout,
        n_trials=n_trials,
        show_progress_bar=show_progress_bar,
    )
    support = np.where(
        objective.best_model.model.feature_importances_ == 0, False, True
    )

    if sum([1 if x else 0 for x in support]) == len(support):
        selector = SelectFromModel(estimator=objective.best_model.model).fit(
            X_train, y_train
        )
        support = selector.get_support()

    return support
