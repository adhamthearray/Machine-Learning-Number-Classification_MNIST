import numpy as np



class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None
        
    def fit(self, x):
        self.mean = np.mean(x, axis=0)
        x = x - self.mean
        
        cov = np.cov(x.T)
        
        eigenValues, eigenVectors = np.linalg.eigh(cov)
        eigenVectors = eigenVectors.T
        
        idxs = np.argsort(eigenValues)[::-1]
        eigenValues = eigenValues[idxs]
        eigenVectors = eigenVectors[idxs]
        
        self.components = eigenVectors[:self.n_components]
        
    
    def transform(self, x):
        x = x - self.mean
        return np.dot(x, self.components.T)