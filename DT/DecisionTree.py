import numpy as np
import math


class DecisionTree:

    class Node:
        def __init__(self, feature=None, threshold=None, left=None, right=None, decision=None):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.decision = decision


    def __init__(self,
                 maxDepth=None,
                 minSamplesSplit=2,
                 criterion="gini",
                 maxFeatures=None,
                 minSampleLeafs=1,
                 classWeights=None,
                 randomState=None):

        self.maxDepth = maxDepth
        self.minSamplesSplit = minSamplesSplit
        self.minSampleLeafs = minSampleLeafs
        self.criterion = criterion
        self.maxFeatures = maxFeatures
        self.allowedCriteria = ["entropy", "gini"]
        self.classWeights = classWeights

        # NEW: deterministic randomness
        self.randomState = randomState
        self.rng = np.random.default_rng(randomState)

        if criterion not in self.allowedCriteria:
            raise ValueError("Criterion has to be either entropy or gini")

        self.root = None


    def majorityLabel(self, labels):
        counts = np.bincount(labels)
        return np.argmax(counts)


    def getThreshold(self, featureValues, labels):

        entropy = []
        labelCount = len(labels)

        sortedIndices = np.argsort(featureValues)
        sortedFeatureValues = featureValues[sortedIndices]
        sortedLabels = labels[sortedIndices]

        leftLabelCounts = np.zeros(np.max(labels) + 1, dtype=int)
        rightLabelCounts = np.bincount(sortedLabels,
                                      minlength=np.max(labels) + 1)

        thresholds = []

        for i in range(len(sortedLabels) - 1):

            leftLabelCounts[sortedLabels[i]] += 1
            rightLabelCounts[sortedLabels[i]] -= 1

            if np.sum(leftLabelCounts) < self.minSampleLeafs:
                continue

            if np.sum(rightLabelCounts) < self.minSampleLeafs:
                continue

            if sortedFeatureValues[i] == sortedFeatureValues[i + 1]:
                continue

            w_entropy = self.computeImpurity(
                leftLabelCounts,
                rightLabelCounts,
                labelCount
            )

            threshold = (
                sortedFeatureValues[i] +
                sortedFeatureValues[i + 1]
            ) / 2

            thresholds.append(threshold)
            entropy.append(w_entropy)

        if len(entropy) == 0:
            return None, None

        minEntropyIndex = np.argmin(entropy)

        return thresholds[minEntropyIndex], entropy[minEntropyIndex]


    def computeImpurity(self,
                        leftLabelCounts,
                        rightLabelCounts,
                        labelCount):

        leftEntropy = 0
        rightEntropy = 0

        if self.classWeights is not None:

            weightsArr = np.array([
                self.classWeights.get(i, 1)
                for i in range(leftLabelCounts.size)
            ])

        else:

            weightsArr = np.ones(leftLabelCounts.size)

        weightedCountsL = weightsArr * leftLabelCounts
        weightedCountsR = weightsArr * rightLabelCounts

        wLeftLabelCount = np.sum(weightedCountsL)
        wRightLabelCount = np.sum(weightedCountsR)

        totalWeight = wLeftLabelCount + wRightLabelCount


        for count in weightedCountsL:

            if count == 0:
                continue

            prob = count / wLeftLabelCount

            if self.criterion == "entropy":
                leftEntropy += -prob * np.log2(prob)
            else:
                leftEntropy += prob ** 2


        for count in weightedCountsR:

            if count == 0:
                continue

            prob = count / wRightLabelCount

            if self.criterion == "entropy":
                rightEntropy += -prob * np.log2(prob)
            else:
                rightEntropy += prob ** 2


        if self.criterion == "entropy":

            w_entropy = (
                (wLeftLabelCount / totalWeight) * leftEntropy +
                (wRightLabelCount / totalWeight) * rightEntropy
            )

        else:

            w_entropy = (
                (wLeftLabelCount / totalWeight) * (1 - leftEntropy) +
                (wRightLabelCount / totalWeight) * (1 - rightEntropy)
            )

        return w_entropy


    def getFeature(self, trainingData, labels):

        entropy = []
        thresholds = []
        featureIndices = []

        noOfFeatures = trainingData.shape[1]


        if self.maxFeatures is not None:

            current_max = self.maxFeatures

            if current_max == "sqrt":
                current_max = int(math.sqrt(noOfFeatures))

            elif current_max == "log2":
                current_max = int(np.log2(noOfFeatures))


            featureArr = self.rng.choice(
                noOfFeatures,
                size=current_max,
                replace=False
            )

        else:

            featureArr = np.arange(noOfFeatures)


        for j in featureArr:

            featureValues = trainingData[:, j]

            optimalThreshold, minEntropy = self.getThreshold(
                featureValues,
                labels
            )

            if optimalThreshold is None:
                continue

            entropy.append(minEntropy)
            thresholds.append(optimalThreshold)
            featureIndices.append(j)


        if len(entropy) == 0:
            return None, None


        minEntropyIndex = np.argmin(entropy)

        return (
            featureIndices[minEntropyIndex],
            thresholds[minEntropyIndex]
        )


    def buildTree(self, data, labels, depth=0):

        if len(np.unique(labels)) == 1:
            return self.Node(decision=np.unique(labels)[0])

        if self.maxDepth is not None and depth >= self.maxDepth:
            return self.Node(decision=self.majorityLabel(labels))

        if len(labels) < self.minSamplesSplit:
            return self.Node(decision=self.majorityLabel(labels))


        feature, threshold = self.getFeature(data, labels)

        if feature is None:
            return self.Node(decision=self.majorityLabel(labels))


        leftMask = data[:, feature] <= threshold
        rightMask = ~leftMask


        leftChild = self.buildTree(
            data[leftMask],
            labels[leftMask],
            depth + 1
        )

        rightChild = self.buildTree(
            data[rightMask],
            labels[rightMask],
            depth + 1
        )


        return self.Node(
            feature,
            threshold,
            leftChild,
            rightChild
        )


    def fit(self, trainingData, labels):

        self.root = self.buildTree(trainingData, labels)


    def traverseTree(self, dataPoint, node):

        if node.decision is not None:
            return node.decision

        if dataPoint[node.feature] <= node.threshold:
            return self.traverseTree(dataPoint, node.left)

        return self.traverseTree(dataPoint, node.right)


    def predict(self, datapoints):

        if self.root is None:
            raise ValueError(
                "Decision tree has not been trained. Call fit() first."
            )

        return np.array([
            self.traverseTree(point, self.root)
            for point in datapoints
        ])