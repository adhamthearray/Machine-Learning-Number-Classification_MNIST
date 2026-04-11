import numpy as np
class DecisionTree:
    class Node:
        def __init__(self , feature = None , threshold = None , left = None , right = None ,decision= None):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.decision = decision

            
    def __init__(self , maxDepth , minSamplesSplit):
        self.maxDepth = maxDepth
        self.minSamplesSplit = minSamplesSplit
        self.root = None
    
    def majorityLabel(self, labels):
        counts = np.bincount(labels)
        return np.argmax(counts)
    def getThreshold(self ,featureValues , labels):
        entropy = []
        labelCount = len(labels)
        sortedIndices = np.argsort(featureValues)
        sortedFeatureValues = featureValues[sortedIndices]
        sortedLabels = labels[sortedIndices]
        leftLabelCounts = np.zeros(np.max(labels)+1 , dtype=int)
        rightLabelCounts =np.bincount(sortedLabels , minlength=np.max(labels)+1)
        thresholds = []
        for i in range(len(sortedLabels)-1):
            leftLabelCounts[sortedLabels[i]]+=1
            rightLabelCounts[sortedLabels[i]]-=1
            if sortedFeatureValues[i] == sortedFeatureValues[i+1]: continue
            w_entropy = self.computeEntropy(leftLabelCounts , rightLabelCounts , labelCount)
            threshold = (sortedFeatureValues[i] + sortedFeatureValues[i+1])/2
            thresholds.append(threshold)
            entropy.append(w_entropy)
        if len(entropy) == 0:
            return None , None
        minEntropyIndex = np.argmin(entropy)
        minEntropy = entropy[minEntropyIndex]
        minThreshold = thresholds[minEntropyIndex]
        return minThreshold , minEntropy
    def computeEntropy(self , leftLabelCounts , rightLabelCounts , labelCount):
            w_entropy = 0
            leftEntropy = 0 
            rightEntropy = 0
            leftLabelCount = np.sum(leftLabelCounts)
            rightLabelCount = np.sum(rightLabelCounts)
            for count in leftLabelCounts:
                if count == 0: continue
                prob = count/leftLabelCount
                leftEntropy += -prob*np.log2(prob)
            for count in rightLabelCounts:
                if count == 0: continue
                prob = count/rightLabelCount
                rightEntropy += -prob*np.log2(prob)
            w_entropy = leftLabelCount/labelCount * leftEntropy +rightLabelCount/labelCount *rightEntropy
            return w_entropy
    def getFeature(self,trainingData , labels):
        entropy = []
        thresholds = []
        featureIndices = []
        noOfFeatures = trainingData.shape[1]
        for j in range(noOfFeatures):
            featureValues = trainingData[:,j]
            optimalThreshold , minEntropy = self.getThreshold(featureValues , labels)
            if optimalThreshold == None: continue
            entropy.append(minEntropy)
            thresholds.append(optimalThreshold)
            featureIndices.append(j)
        minEntropyIndex = np.argmin(entropy)
        return featureIndices[minEntropyIndex] , thresholds[minEntropyIndex]
    def buildTree(self , data , labels , depth = 0 ):
        #Base case law el labels el da5laly kolaha nafs el labels aw el depth aw el min sample leaves bas for now labels bas
        if len(np.unique(labels)) == 1:
            decisionClass = (np.unique(labels))[0]
            return self.Node(decision=decisionClass)
        if depth >= self.maxDepth:
            return self.Node(decision=self.majorityLabel(labels))
        if len(labels) < self.minSamplesSplit:
            return self.Node(decision=self.majorityLabel(labels))
        
        feature , threshold  = self.getFeature(data , labels)
        #Case if all values are the same and there is no valid split
        if feature == None and threshold == None:
            return self.Node(decision=self.majorityLabel(labels))
       
        leftData = data[data[:,feature]<= threshold]
        leftLabels = labels[data[:,feature]<= threshold]
        rightData = data[data[:,feature]> threshold]
        rightLables = labels[data[:,feature] >  threshold]
        leftChild = self.buildTree(leftData ,leftLabels , depth+1)
        rightChild = self.buildTree(rightData , rightLables , depth+1)
        return self.Node(feature , threshold , leftChild , rightChild)

    def fit(self , trainingData , labels):
        self.root = self.buildTree(trainingData , labels)
    def traverseTree(self , dataPoint , node):
        if node.decision != None:
            return node.decision
        if dataPoint[node.feature] <= node.threshold:
            return self.traverseTree(dataPoint , node.left)
        else:
            return self.traverseTree(dataPoint , node.right)
    def predict(self , datapoints):
        if self.root is None:
            raise ValueError("Decision tree has not been trained. Call fit() first.")
        predictions = []
        for datapoint in datapoints:
            predictions.append(self.traverseTree(datapoint , self.root))
        return np.array(predictions)