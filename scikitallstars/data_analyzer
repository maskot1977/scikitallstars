import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

class SplitTester:
    def __init__(self, 
                 test_size = 0.1, n_trials = 10,
                 smallest = 0, largest = 100, num_seeds = 10, 
                 verbose = True):
        self.test_size = test_size
        self.n_trials = n_trials
        self.smallest = smallest
        self.largest = largest
        self.num_seeds = num_seeds
        self.regressor = RandomForestRegressor(n_jobs=-1)
        self.classifier = RandomForestClassifier(n_jobs=-1)
        self.best_seed = None
        self.best_score = None
        self.history = []
        self.verbose = verbose
        self.feature_importances = []
        self.feature_names = []

    def __call__(self, X, Y):
        X = pd.DataFrame(X)
        Y = pd.DataFrame(Y).iloc[:, 0]
        self.feature_names = X.columns
        if len(list(set(list(Y)))) == 2:
            self.is_regressor = False
            self.model = self.classifier
        else:
            self.is_regressor = True
            self.model = self.regressor

        for random_state in np.random.randint(
            self.smallest, 
            self.largest, 
            self.num_seeds
            ):
            X_train, X_test, Y_train, Y_test = train_test_split(
                X, Y, 
                random_state=random_state, 
                test_size=self.test_size
            )
            for _ in range(self.n_trials):
                self.model.fit(X_train, Y_train)
                self.feature_importances.append([random_state, self.model.feature_importances_])
                score = self.model.score(X_test, Y_test)
                if self.verbose:
                    print([random_state, score])
                self.history.append([random_state, score])
                if self.best_seed is None or self.best_score < score:
                    self.best_seed = random_state
                    self.best_score = score

        return self.best_seed

    def depict_boxplot(self):
        data = pd.DataFrame(self.history)
        data.columns = ["seed", "score"]
        ax = data.boxplot(column="score", by="seed")
        ax.set_title("")
        ax.set_ylabel("score (test)")

    def depict_feature_importances(self, top_n=10):
        importances = [list(fi) for random_state, fi in tester.feature_importances]
        mean_importances = pd.DataFrame(importances).describe().T['mean'].values
        n_shown = 0
        for i in list(np.argsort(mean_importances))[::-1]:
            tmp_ary = []
            for random_state, fi in tester.feature_importances:
                tmp_ary.append([random_state, fi[i]])
                       
            data = pd.DataFrame(tmp_ary)
            data.columns = ["seed", "score"]
            ax = data.boxplot(column="score", by="seed")
            ax.set_title(self.feature_names[i])
            ax.set_ylabel("feature importance")
            n_shown += 1
            if n_shown >= top_n:
                break 
