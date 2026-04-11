import numpy as np
class DecisionTree:
    class Node:
        def __init__(self , feature = None , threshold = None , left = None , right = None ,decision= None):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.decision = decision

            
    def __init__(self , maxDepth):
        self.maxDepth = maxDepth
        self.root = None
    
    
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
        noOfFeatures = trainingData.shape[1]
        for j in range(noOfFeatures):
            featureValues = trainingData[:,j]
            optimalThreshold , minEntropy = self.getThreshold(featureValues , labels)
            entropy.append(minEntropy)
            thresholds.append(optimalThreshold)
        minEntropyIndex = np.argmin(entropy)
        return minEntropyIndex , thresholds[minEntropyIndex]

            








    