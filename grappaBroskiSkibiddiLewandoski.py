import numpy as np
from DT.DecisionTree import DecisionTree
#THhi is still under test and not safe and might not disccuss it until we check some things
class HobaTitoMambo:
    
    def __init__(self, n_trees=10, max_depth=10, max_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []

    def __bootstrap_samples(self, x, y):
        n_samples = x.shape[0]
        idxs = np.random.choice(n_samples, size=n_samples, replace=True)
        return x[idxs], y[idxs]

    def fit(self, x, y):
        self.trees = []

        for i in range(self.n_trees):
            dt = DecisionTree(
                maxDepth=self.max_depth,
                maxFeatures=self.max_features
            )

            x_s, y_s = self.__bootstrap_samples(x, y)
            dt.fit(x_s, y_s)
            self.trees.append(dt)

    def predict(self, x_test):
        preds = np.array([tree.predict(x_test) for tree in self.trees])
        preds = preds.T

        res = np.zeros(len(x_test), dtype=int)

        for i, sample_votes in enumerate(preds):
            vals, counts = np.unique(sample_votes, return_counts=True)
            res[i] = vals[np.argmax(counts)]

        return res