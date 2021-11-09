import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


class KMeansSplitter:
    def __init__(self, representative=True, test_size=0.1, random_state=None):
        self.representative = representative
        self.test_size = test_size
        self.random_state = random_state
        self.error = 0.0001
        self.max_trial = 530000

    def __call__(self, X):
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
