from sklearn.neighbors import NearestNeighbors
from sklearn.svm import OneClassSVM


class KNN:
    def __init__(self, n_neighbors=5, out=0.05, algorithm="ball_tree"):
        self.n_neighbors = n_neighbors
        self.out = out
        self.model = False
        self.algorithm = algorithm
        self.distances = False
        self.indices = False
        self.threshold = False
        self.len_data = False

    def fit(self, X):
        self.len_data = len(X)
        self.model = NearestNeighbors(
            n_neighbors=self.n_neighbors, algorithm=self.algorithm
        ).fit(X)
        self.distances, self.indices = self.model.kneighbors(X)
        self.threshold = sorted(self.distances[:, self.n_neighbors - 1])[
            int((self.len_data - 1) * (1 - self.out))
        ]

    def transform(self, x):
        self.distances, self.indices = self.model.kneighbors(x)
        return self.distances[:, self.n_neighbors - 1]

    def transform_bin(self, x):
        self.transform(x)
        self.Z = self.distances[:, self.n_neighbors - 1]
        return np.where(self.Z >= self.threshold, 0, 1)


class OCSVM:
    def __init__(self, out=0.05):
        self.model = OneClassSVM()
        self.out = out
        self.threshold = False
        self.len_data = False

    def fit(self, X):
        self.len_data = len(X)
        self.model.fit(X)
        self.threshold = sorted(self.model.decision_function(X))[
            int((self.len_data - 1) * self.out)
        ]

    def transform(self, x):
        return self.model.decision_function(x)

    def transform_bin(self, x):
        self.Z = self.model.decision_function(x)
        return np.where(self.Z >= self.threshold, 1, 0)
