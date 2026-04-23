import numpy as np

class LogReg():
    def __init__(self, max_iterations=100, threshold=0.5, learning_rate=0.01, class_weight=None, reg_eqn=None, reg_param=0, random_state=42):
        self.threshold = threshold
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.class_weight = class_weight
        self.reg_eqn = reg_eqn
        self.reg_param=reg_param
        self.random_state=random_state
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
        np.random.seed(42)
        num_samples, num_features = x.shape
        classes = np.unique(y)
        self.weights = np.random.randn(num_features) * 0.01
        self.bias = 0.0
        
        
        classes_weights = {}
        if self.class_weight is None:
            classes_weights = {label: 1.0 for label in classes}
        elif self.class_weight == 'balanced':
            for label in classes:
                classes_weights[label] = num_samples / (len(classes) * np.sum(y == label))
        else:
            classes_weights = self.class_weight
            
        loss_weights = np.ones_like(y, dtype=float)
        for label in classes:
            loss_weights[y == label] = classes_weights[label]
            
        weight_sum = np.sum(loss_weights)
            
        for i in range(self.max_iterations):
            linear_predictions = self.__linear(x)
            probas = self.__sigmoid(linear_predictions)
            cost = self.__loss(probas, y)
            
            if i % 10 == 0:
                print(f"Iteration {i}, Loss: {cost}")
            
            
            dw = np.dot(x.T, loss_weights*(probas-y)) / weight_sum
            db = np.sum(loss_weights*(probas-y)) / weight_sum
            
            if self.reg_eqn == 'L2':
                dw += (2 * self.reg_param / weight_sum) * self.weights
            elif self.reg_eqn == 'L1':
                dw += (self.reg_param / weight_sum) * np.sign(self.weights)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, test):
        linear_predictions = self.__linear(test)
        probas = self.__sigmoid(linear_predictions)
        return (probas >= self.threshold).astype(int)




class MultiLogReg():
    def __init__(self, max_iterations=100, learning_rate=0.01, class_weight=None, reg_eqn=None, reg_param=0, random_state=42):
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.class_weight = class_weight
        self.reg_eqn = reg_eqn
        self.reg_param = reg_param
        self.random_state = random_state
        self.classes = None
        self.weights = None
        self.bias = None
        
    def __linear(self, x):
        return np.dot(x, self.weights) + self.bias
    
    def __softmax(self, z):
        e_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
        return e_z / np.sum(e_z, axis=-1, keepdims=True)
    
    def __one_hot(self, y):
        n_samples = len(y)
        classes = np.unique(y)
        one_hot = np.zeros((n_samples, len(classes)))
        one_hot[np.arange(n_samples), y] = 1
        return one_hot
    
    def __loss(self, preds, y_one_hot):
        m = len(y_one_hot)
        eps = 1e-9
        preds = np.clip(preds, eps, 1 - eps)
        return -np.sum(y_one_hot * np.log(preds)) / m
    
    def fit(self, x, y):
        np.random.seed(self.random_state)
        num_samples, num_features = x.shape
        self.classes = np.unique(y)
        num_classes = len(self.classes)
        self.weights = np.random.randn(num_features, num_classes) * 0.01
        self.bias = np.zeros(num_classes)
        
        classes_weights = {}
        if self.class_weight is None:
            classes_weights = {label: 1.0 for label in self.classes}
        elif self.class_weight == 'balanced':
            for label in self.classes:
                classes_weights[label] = num_samples / (len(self.classes) * np.sum(y == label))
        else:
            classes_weights = self.class_weight
            
        loss_weights = np.ones_like(y, dtype=float)
        for label in self.classes:
            loss_weights[y == label] = classes_weights[label]
            
        weight_sum = np.sum(loss_weights)
        
        loss_weights = loss_weights.reshape(-1,1)
        y_one_hot = self.__one_hot(y)
            
        for i in range(self.max_iterations):
            linear_predictions = self.__linear(x)
            probas = self.__softmax(linear_predictions)
            cost = self.__loss(probas, y_one_hot)
            
            if i % 10 == 0:
                print(f"Iteration {i}, Loss: {cost}")
            
            dw = np.dot(x.T, loss_weights*(probas-y_one_hot)) / weight_sum
            db = np.sum(loss_weights*(probas-y_one_hot), axis=0) / weight_sum
            
            if self.reg_eqn == 'L2':
                dw += (2 * self.reg_param / weight_sum) * self.weights
            elif self.reg_eqn == 'L1':
                dw += (self.reg_param / weight_sum) * np.sign(self.weights)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, test):
        linear_predictions = self.__linear(test)
        probas = self.__softmax(linear_predictions)
        preds_indices = np.argmax(probas, axis=1)
        return self.classes[preds_indices]