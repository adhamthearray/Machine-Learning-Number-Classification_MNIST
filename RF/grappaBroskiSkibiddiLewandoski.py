import numpy as np
from DT.DecisionTree import DecisionTree

class HobaTitoMambo:
    
    def __init__(self, n_trees=10, max_depth=10, max_features="sqrt", min_samples_split=2, min_sample_leafs=1, criterion="gini",random_state=None):
        
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_sample_leafs = min_sample_leafs
        self.criterion = criterion
        self.random_state = random_state
        self.trees = []
        self.rng = np.random.default_rng(self.random_state)

    def __bootstrap_samples(self, x, y):
        n_samples = x.shape[0]
        idxs = self.rng.choice(n_samples, size=n_samples, replace=True)
        return x[idxs], y[idxs]

    def fit(self, x, y):
        self.trees = []

        for i in range(self.n_trees):
            tree_seed = self.random_state + i if self.random_state is not None else None
            
            dt = DecisionTree(maxDepth=self.max_depth,minSamplesSplit=self.min_samples_split,criterion=self.criterion,maxFeatures=self.max_features,minSampleLeafs=self.min_sample_leafs,randomState=tree_seed)

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