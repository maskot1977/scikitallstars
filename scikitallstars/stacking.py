import pandas as pd
from sklearn.ensemble import StackingClassifier, StackingRegressor

class StackingObjective:
    def __init__(self, objective, X_train, y_train, test_size=0.1, verbose=True, train_random_state=None):
        self.x_train = X_train
        self.y_train = y_train
        self.verbose = verbose
        self.objective = objective
        self.best_score = None
        self.best_model = None
        self.already_tried = {}
        self.rf_max_depth = [2, 32]
        self.rf_max_features = ["auto", "sqrt", "log2"]
        self.rf_n_estimators = [50, 200]
        self.rf_warm_start = [True, False]
        self.rf_criterion = ["mse", "mae"]
        self.rf_oob_score = [True, False]
        self.n_trial = 0
        self.support = objective.support
        self.is_regressor = objective.is_regressor
        self.test_size = test_size,
        self.train_random_state = train_random_state

    def __call__(self, trial):
        self.n_trial += 1
        estimators = []
        key = ""
        for model_name in self.objective.get_model_names():
            if model_name in self.objective.best_models.keys():
                in_out = trial.suggest_int(model_name, 0, 1)
                key += str(in_out)
                if in_out == 1:
                    estimators.append(
                        (model_name, self.objective.best_models[model_name].model)
                    )

        params = {}
        params["n_estimators"] = trial.suggest_int(
            "rf_n_estimators", self.rf_n_estimators[0], self.rf_n_estimators[1]
        )
        # params["max_features"] = trial.suggest_categorical(
        #            "rf_max_features", self.rf_max_features
        #        )
        params["n_jobs"] = -1
        params["warm_start"] = trial.suggest_categorical(
            "rf_warm_start", self.rf_warm_start
        )
        params["max_depth"] = trial.suggest_int(
            "rf_max_depth", self.rf_max_depth[0], self.rf_max_depth[1]
        )
        # params["criterion"] = trial.suggest_categorical(
        #            "rf_criterion", self.rf_criterion
        #        )
        params["oob_score"] = trial.suggest_categorical(
            "rf_oob_score", self.rf_oob_score
        )

        if key in self.already_tried.keys():
            pass
            # return self.already_tried[key]

        if len(estimators) == 0:
            return 0 - 530000

        if True:  # self.support is None:
            x_train, x_test, y_train, y_test = train_test_split(
                self.x_train, self.y_train, test_size=self.test_size, random_state=self.train_random_state
            )
        else:
            x_train, x_test, y_train, y_test = train_test_split(
                self.x_train.iloc[:, self.support], self.y_train, test_size=self.test_size
            )
        stacking_model1 = stacking(
            self.objective, estimators=estimators, verbose=self.verbose, params=params
        )
        stacking_model1.support = self.objective.support
        stacking_model1.fit(x_train, y_train)
        score = stacking_model1.score(x_test, y_test)
        if self.verbose:
            print("Trial ", self.n_trial)
            print(score)
            print(stacking_model1.final_estimator_)

        if self.best_score is None:
            self.best_score = score
            self.best_model = stacking_model1
        elif self.best_score < score:
            self.best_score = score
            self.best_model = stacking_model1

        self.already_tried[key] = score

        return score

    def predict(self, X):
        return self.best_model.predict(X)

    def score(self, X, Y):
        return self.best_model.score(X, Y)


def get_best_stacking(
    objective,
    X_train,
    y_train,
    verbose=True,
    timeout=1000,
    n_trials=50,
    show_progress_bar=True,
):
    X_train = pd.DataFrame(X_train)
    if type(y_train) is not pd.core.series.Series:
        y_train = pd.DataFrame(y_train)[0]
    stacking_objective = StackingObjective(objective, X_train, y_train)
    study = optuna.create_study(direction="maximize")

    try_all = {}
    for model_name in objective.get_model_names():
        try_all[model_name] = 1
    study.enqueue_trial(try_all)

    threshold = sum(
        [objective.best_scores[name] for name, model in objective.best_models.items()]
    ) / len(objective.best_models.items())
    try_threshold = {}
    for model_name in objective.get_model_names():
        if model_name in objective.best_models.keys():
            model = objective.best_models[model_name]
            if objective.best_scores[model_name] >= threshold:
                try_threshold[model_name] = 1
            else:
                try_threshold[model_name] = 0
    study.enqueue_trial(try_threshold)

    study.optimize(
        stacking_objective,
        timeout=timeout,
        n_trials=n_trials,
        show_progress_bar=show_progress_bar,
    )
    return stacking_objective


class StackingRegressorS(StackingRegressor):
    def __init__(self, **args):
        super(StackingRegressor, self).__init__(**args)
        self.support = None
        self.x = None

    def fit(self, x, y):
        if self.support is None or len(self.support) != x.shape[1]:
            return super(StackingRegressor, self).fit(x, y)
        else:
            return super(StackingRegressor, self).fit(x.iloc[:, self.support], y)

    def score(self, x, y):
        x = pd.DataFrame(x)
        if type(y) is not pd.core.series.Series:
            try:
                y = pd.DataFrame(y)[0]
            except:
                pass
        if self.support is None or len(self.support) != x.shape[1]:
            return super(StackingRegressor, self).score(x, y)
        else:
            return super(StackingRegressor, self).score(x.iloc[:, self.support], y)

    def predict(self, x):
        x = pd.DataFrame(x)
        if self.support is None or len(self.support) != x.shape[1]:
            return super(StackingRegressor, self).predict(x)
        else:
            return super(StackingRegressor, self).predict(x.iloc[:, self.support])


class StackingClassifierS(StackingClassifier):
    def __init__(self, **args):
        super(StackingClassifier, self).__init__(**args)
        self.support = None
        self.classes_ = [0, 1]

    # @property
    # def classes_(self):
    #   return self.classes_

    def fit(self, x, y):
        if self.support is None or len(self.support) != x.shape[1]:
            return super(StackingClassifier, self).fit(x, y)
        else:
            return super(StackingClassifier, self).fit(x.iloc[:, self.support], y)

    def score(self, x, y):
        x = pd.DataFrame(x)
        y = pd.DataFrame(y)[0]
        if self.support is None or len(self.support) != x.shape[1]:
            return super(StackingClassifier, self).score(x, y)
        else:
            return super(StackingClassifier, self).score(x.iloc[:, self.support], y)

    def predict(self, x):
        x = pd.DataFrame(x)
        if self.support is None or len(self.support) != x.shape[1]:
            return super(StackingClassifier, self).predict(x)
        else:
            return super(StackingClassifier, self).predict(x.iloc[:, self.support])


def stacking(
    objective,
    final_estimator=None,
    use_all=False,
    verbose=True,
    estimators=None,
    params=None,
):
    if estimators is None:
        if use_all:
            estimators = [
                (name, model.model) for name, model in objective.best_models.items()
            ]

        else:
            threshold = sum(
                [
                    objective.best_scores[name]
                    for name, model in objective.best_models.items()
                ]
            ) / len(objective.best_models.items())
            estimators = []
            for name, model in objective.best_models.items():
                if objective.best_scores[name] >= threshold:
                    estimators.append((name, model.model))

    if verbose:
        print([name for name, model in estimators])

    if objective.is_regressor:
        if final_estimator is None:
            if params is None:
                final_estimator = RandomForestRegressor()
            else:
                final_estimator = RandomForestRegressor(**params)

        model = StackingRegressorS(
            estimators=estimators,
            final_estimator=final_estimator,
        )
        model.support = objective.support
    else:
        if final_estimator is None:
            if params is None:
                final_estimator = RandomForestClassifier()
            else:
                final_estimator = RandomForestClassifier(**params)

        model = StackingClassifierS(
            estimators=estimators,
            final_estimator=final_estimator,
        )
        model.support = objective.support
    return model
