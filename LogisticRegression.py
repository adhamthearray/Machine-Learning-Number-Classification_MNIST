import numpy as np

class LogReg():
    def __init__(self, max_iterations=100, threshold=0.5, learning_rate=0.01):
        self.threshold = threshold
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = 0.0
        
    def __linear(self, x):
        return np.dot(x, self.weights) + self.bias
    
    def __sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))
    
    def __loss(self, preds, y):
        m = len(y)
        eps = 1e-9
        preds = np.clip(preds, eps, 1 - eps)
        return - (1/m) * np.sum(y * np.log(preds) + (1 - y) * np.log(1 - preds))
    
    def fit(self, x, y):
        num_samples, num_features = x.shape
        self.weights = np.zeros(num_features)
        self.bias = 0.0
        
        for i in range(self.max_iterations):
            linear_predictions = self.__linear(x)
            probas = self.__sigmoid(linear_predictions)
            cost = self.__loss(probas, y)
            
            if i % 10 == 0:
                print(f"Iteration {i}, Loss: {cost}")
            
            dw = np.dot(x.T, (probas-y)) / num_samples
            db = np.sum(probas-y) / num_samples
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, test):
        linear_predictions = self.__linear(test)
        probas = self.__sigmoid(linear_predictions)
        return (probas >= self.threshold).astype(int)


'''
def main():
    X3 = np.random.randn(100, 2)
    y3 = (X3[:, 0] + X3[:, 1] > 0).astype(int)
    
    split = 80

    X_train = X3[:split]
    y_train = y3[:split]

    X_test = X3[split:]
    y_test = y3[split:]
    
    model4 = LogReg(learning_rate=0.1, max_iterations=1000)
    model4.fit(X_train, y_train)

    preds4 = model4.predict(X_test)
    print("Test Accuracy:", np.mean(preds4 == y_test))
    
    
main()
'''