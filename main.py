import csv
import numpy as np
import random
import math
from sklearn.model_selection import KFold


def readFile(fileName):
    lines = csv.reader(open(fileName, "r"))
    inputData = list(lines)
    for i in range(len(inputData)):
        inputData[i] = [float(x) for x in inputData[i]]
    return inputData

def getKFoldCrossValidationSplit(dataSet, numberSplits):
    kf = KFold(n_splits=numberSplits)
    kf.get_n_splits(dataSet)
    arrayDataSet = np.array(dataSet)
    for train_index, test_index in kf.split(dataSet):
        #print("TRAIN:", train_index, "TEST:", test_index)
        trainingData, testData = arrayDataSet[train_index], arrayDataSet[test_index]
        return trainingData, testData

def getRandomFraction(dataSet, fraction):
    trainingData = []
    testData = dataSet
    trainingDataSize = int(len(dataSet) * fraction)
    while len(trainingData) < trainingDataSize:
        index = random.randrange(len(testData))
        trainingData.append(testData[index])
    # print(len(trainingData))
    return trainingData, testData

class NaiveBayesClassifier:

    def __init__(self):
        self = self

    def getLabelFeaturesMap(self, dataSet):
        featureClassMap = {}
        for i in range(len(dataSet[0])):
            if (dataSet[0][i][-1] not in featureClassMap):
                featureClassMap[dataSet[0][i][-1]] = []
            featureClassMap[dataSet[0][i][-1]].append(dataSet[0][i][:-1])
        # print(featureClassMap)
        return featureClassMap

    def calculateVariance(self, values):
        avg = sum(values) / float(len(values))
        variance = sum([pow(x - avg, 2) for x in values]) / float(len(values) - 1)
        return variance

    def calculateGaussianPDF(self, mean, variance, value):
        exponent = math.exp(-(math.pow(value - mean, 2) / (2 * variance)))
        return (1 / (math.sqrt(2 * math.pi) * math.sqrt(variance))) * exponent

    def getClassLabelProbability(self, testDataRecord, stats):
        probabilityForClassLabel = {}
        for classLabel, groupStatistics in stats.items():
            probabilityForClassLabel[classLabel] = 1
            for i in range(len(groupStatistics)):
                mean, variance = groupStatistics[i]
                value = testDataRecord[i]
                probabilityForClassLabel[classLabel] *= self.calculateGaussianPDF(mean, variance, value)
        return probabilityForClassLabel

    def getStatistics(self, dataset):
        stats = {}
        classFeatureMap = self.getLabelFeaturesMap(dataset)
        for classLabel, features in classFeatureMap.items():
            stats[classLabel] = [(sum(attribute)/float(len(attribute)), self.calculateVariance(attribute)) for attribute in zip(*features)]
        return stats

    def predictClassLabel(self, input, stats):
        probabilitiesMap = self.getClassLabelProbability(input, stats)
        prob = -1
        label = None
        for classLabel, probability in probabilitiesMap.items():
            if label is None or probability > prob:
                prob = probability
                label = classLabel
        return label

    def getPredictionsUsingNB(self, testSet, stats):
        predictions = []
        for i in range(len(testSet)):
            result = self.predictClassLabel(testSet[i], stats)
            predictions.append(result)
        return predictions

    def getAccuracy(self, testSet, predictions):
        count = 0
        for i in range(len(testSet)):
            if testSet[i][-1] == predictions[i]:
                count += 1
        return (count / float(len(testSet))) * 100.0

class LogisticRegression:

    def __init__(self):
        self = self

    def linear_forward(self, X, W, W_not):
        return (np.dot(X, W.T) + W_not)

    def sigmoid(self, Z):
        return 1 / (1 + np.exp((-1)*(Z)))

    def sigmoid_derivative(self, Z):
        return self.sigmoid(Z) * (1 - self.sigmoid(Z))

    def cost_derivative(self, X, Y, W, W_not):
        return Y - self.sigmoid(self.linear_forward(X,W,W_not))

    def trainModel(self, X, Y, W, W_not, eta):
        for i in range(300):
            grad = self.cost_derivative(X,Y,W,W_not) * self.sigmoid_derivative(self.linear_forward(X,W,W_not))
            W_not = W_not + (1/len(X)) * eta * np.sum(grad, axis=0)
            W = W + (1/len(X)) * eta * np.sum(X*grad, axis=0)
        return W, W_not

def main():
    fileName = './abc.csv'
    dataSet = readFile(fileName)

    nbModel = NaiveBayesClassifier()
    lrModel = LogisticRegression()

    kf = KFold(n_splits=3)
    kf.get_n_splits(dataSet)
    arrayDataSet = np.array(dataSet)
    foldIteration = 0

    fraction = 0.5
    for train_index, test_index in kf.split(dataSet):
        foldIteration += 1
        trainingData, testData = arrayDataSet[train_index], arrayDataSet[test_index]

        accuracyforNB = []
        accuracyforLR = []
        i = 0
        accuracyNBSum = 0
        accuracyLRSum = 0
        while i < 5:
            trainingSetPart = getRandomFraction(trainingData, fraction)
            predictions = nbModel.getPredictionsUsingNB(testData, nbModel.getStatistics(trainingSetPart))
            accuracyNBSum += nbModel.getAccuracy(testData, predictions)
            trainingSetPart, testData = getRandomFraction(trainingData, fraction)
            trainingSetPart = np.array(trainingSetPart)
            [X, Y] = np.split(trainingSetPart, [len(trainingSetPart[0]) - 1], axis=1)
            [Test_X, Test_Y] = np.split(testData, [len(testData[0]) - 1], axis=1)
            W = np.random.randn(1, 4) * 0
            W_not = 0
            eta = 0.3
            lrTrainedW, lrTrainedW_not = lrModel.trainModel(X, Y, W, W_not, eta)
            Pred = lrModel.sigmoid(lrModel.linear_forward(Test_X, lrTrainedW, lrTrainedW_not))
            Pred = np.array([1 if x > 0.5 else 0 for x in Pred])
            Test_Y = np.squeeze(Test_Y)
            lrAccuracyVal = np.sum(Pred == Test_Y) / len(Test_Y)
            accuracyLRSum += lrAccuracyVal * 100
            i += 1
        accuracyforLR.append(accuracyLRSum/5)
        accuracyforNB.append(accuracyNBSum/5)
main()