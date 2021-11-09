import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split


class SplitTester:
    def __init__(
        self,
        test_size=0.1,
        n_trials=5,
        smallest=0,
        largest=20,
        num_seeds=20,
        verbose=True,
    ):
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
        self.feature_importances = {}
        self.feature_names = []
        self.dist_train_train = []
        self.dist_train_test = []
        self.dist_test_test = []
        self.scores = {}

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

        random_states = [x for x in range(self.smallest, self.largest)]
        np.random.shuffle(random_states)
        for i, random_state in enumerate(random_states):
            if i >= self.num_seeds:
                break
            X_train, X_test, Y_train, Y_test = train_test_split(
                X, Y, random_state=random_state, test_size=self.test_size
            )
            for d in cos_sim_dist(X_train.values, X_train.values):
                self.dist_train_train.append([random_state, d])

            for d in cos_sim_dist(X_train.values, X_test.values):
                self.dist_train_test.append([random_state, d])

            for d in cos_sim_dist(X_test.values, X_test.values):
                self.dist_test_test.append([random_state, d])

            for _ in range(self.n_trials):
                self.model.fit(X_train, Y_train)
                if random_state not in self.feature_importances.keys():
                    self.feature_importances[random_state] = []
                self.feature_importances[random_state].append(
                    list(self.model.feature_importances_)
                )
                score = self.model.score(X_test, Y_test)
                if random_state not in self.scores.keys():
                    self.scores[random_state] = []
                self.scores[random_state].append(score)
                if self.verbose:
                    print([i, random_state, score])
                self.history.append([random_state, score])
                if self.best_seed is None or self.best_score < score:
                    self.best_seed = random_state
                    self.best_score = score

        return self.best_seed

    def depict_boxplot(self):
        data = pd.DataFrame(self.history)
        data.columns = ["split seed", "score"]
        ax = data.boxplot(column="score", by="split seed")
        ax.set_title("test_size={}".format(self.test_size))
        plt.suptitle("")
        ax.set_ylabel("score (test)")
        plt.show()

    def depict_feature_importances(self, n_features=10):
        for random_state, fi in self.feature_importances.items():
            data = pd.DataFrame(fi)
            data.columns = self.feature_names
            sorted_idx = list(np.argsort(data.describe().T["mean"].values))[::-1]
            data = data.iloc[:, sorted_idx[:n_features]]
            data.boxplot(vert=False)
            plt.title(
                "random_state={}, score={}".format(
                    random_state,
                    sum(self.scores[random_state]) / len(self.scores[random_state]),
                )
            )
            plt.show()


def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def cos_sim_dist(ary1, ary2):
    dist = []
    for v1 in ary1:
        for v2 in ary2:
            dist.append(cos_sim(v1, v2))
    return dist

