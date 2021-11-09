import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split


class SplitTester:
    def __init__(
        self,
        test_size=0.1,
        n_trials=5,
        num_seeds=20,
        smallest=0,
        largest=20,
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

    def __call__(self, X, Y, splitter=train_test_split):
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
            X_train, X_test, Y_train, Y_test = splitter(
                X, Y, random_state=random_state, test_size=self.test_size
            )
            for d in cos_sim_dist(X_train.values, X_train.values):
                self.dist_train_train.append([random_state, d])

            for d in cos_sim_dist(X_train.values, X_test.values):
                self.dist_train_test.append([random_state, d])

            for d in cos_sim_dist(X_test.values, X_test.values):
                self.dist_test_test.append([random_state, d])

            for _ in range(self.n_trials):
                Y_train = np.ravel(pd.DataFrame(Y_train).values)
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


class KMeansSplitter:
    def __init__(self, representative=True, test_size=0.1, random_state=None):
        self.representative = representative
        self.test_size = test_size
        self.random_state = random_state
        self.error = 0.0001
        self.max_trial = 530000

    def __call__(self, X, Y, test_size=0.1, random_state=None):
        X = pd.DataFrame(X)
        Y = pd.DataFrame(Y)
        self.test_size = test_size
        self.random_state = random_state
        train_ids, test_ids = self.split_ids(X)
        return (
            X.iloc[train_ids, :],
            X.iloc[test_ids, :],
            Y.iloc[train_ids, :],
            Y.iloc[test_ids, :],
        )

    def split_ids(self, X):
        X = pd.DataFrame(X)

        n_clusters = int(X.shape[0] * self.test_size)
        cids = KMeans(
            n_clusters=n_clusters, random_state=self.random_state
        ).fit_predict(X)
        clusters = {}
        for id, cid in enumerate(cids):
            if cid not in clusters.keys():
                clusters[cid] = []
            clusters[cid].append(id)

        train_ids = []
        test_ids = []
        if self.representative:
            n_trial = 0
            while (
                abs(
                    (len(test_ids) + 0.1) / (len(test_ids) + len(train_ids) + 0.1)
                    - self.test_size
                )
                > self.error
            ):
                n_trial += 1
                if n_trial > self.max_trial:
                    break
                train_ids = []
                test_ids = []
                for cid, ids in clusters.items():
                    np.random.shuffle(ids)
                    split_index = int(len(ids) * self.test_size)
                    if split_index == 0 and np.random.rand() <= self.test_size:
                        split_index = 1
                    train_ids += ids[split_index:]
                    test_ids += ids[:split_index]
                self.error += 0.0001

        else:
            n_trial = 0
            while (
                abs(
                    (len(test_ids) + 0.1) / (len(test_ids) + len(train_ids) + 0.1)
                    - self.test_size
                )
                > self.error
            ):
                n_trial += 1
                if n_trial > self.max_trial:
                    break
                train_ids = []
                test_ids = []
                for cid, ids in clusters.items():
                    np.random.shuffle(ids)
                    if np.random.rand() <= self.test_size:
                        test_ids += ids
                    else:
                        train_ids += ids
                self.error += 0.0001

        return train_ids, test_ids


def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def cos_sim_dist(ary1, ary2):
    dist = []
    for v1 in ary1:
        for v2 in ary2:
            dist.append(cos_sim(v1, v2))
    return dist
