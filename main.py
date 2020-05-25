from scipy.stats import ks_2samp
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, RepeatedKFold
from numpy import mean

from src.ksType import ksType
from src.Instance import Instance

import random


# Configuration
dataFile = 'resources/breast-cancer-wisconsin.data'

skipRowsWithInvalidFeature = True   # Skip rows with '?' in features
defaultValueOfInvalidFeature = 1    # Replace '?' with value if skipRowsWithInvalidFeature is False

# Data settings
columnId = 0            # Index of column with id
columnCancerClass = 10  # Index of column with cancer class
columnFirstFeature = 1  # Index of column with first feature

quantityOfFeatures = 9  # Quantity of features in file

# Algorithms settings
kNN = [1, 5, 10]                            # kNN settings
metricTypes = ['euclidean', 'manhattan']    # Distance metrics


def main():
    instances = loadDataFromFile(dataFile, skipRowsWithInvalidFeature = False, defaultValueOfInvalidFeature = 1)
    ranking = kolmogorovTest(instances, quantityOfFeatures)
    # featuresIds = [x.getParamID() for x in ranking]

    # teachingData, testData = divideInstances(instances)

    # for feature in range(0, ranking.__len__()):
        # print(ranking[feature].getParamID() + 1, '\t', ranking[feature].getPValue(), '\t', ranking[feature].getStatistic())

    crossValidation(kNN, metricTypes, instances, ranking)

    # score = kNNAlgorithm(1, teachingData, testData, featuresIds, 'euclidean')
    # print(score)


def loadDataFromFile(fileName, skipRowsWithInvalidFeature = False, defaultValueOfInvalidFeature = 1):
    instances = []

    file = open(fileName, 'r').read()
    lines = file.split('\n')

    for line in lines:
        if '?' in line and skipRowsWithInvalidFeature is True:
            continue

        row = line.split(',')

        if row.__len__() == 11:
            instance = Instance(row[columnId], row[columnFirstFeature:quantityOfFeatures + 1], row[columnCancerClass], defaultValueOfInvalidFeature)
            instances.append(instance)

    return instances


def kolmogorovTest(instances, quantityOfFeatures):
    featuresRanking = []
    dataDir = {
        '2': {},
        '4': {}
    }

    for feature in range(0, quantityOfFeatures):
        dataDir['4'][feature] = []
        dataDir['2'][feature] = []
        for instance in instances:
            dataDir[instance.getCancerType()][feature].append(instance.getFeatureValues()[feature])

        statistic, pValue = ks_2samp(dataDir['4'][feature], dataDir['2'][feature])
        featuresRanking.append(ksType(feature, statistic, pValue))

    return sorted(featuresRanking, key=ksType.getStatistic, reverse=True)

def divideInstances(instances):
    teachingData = []
    testData = []
    instancesNumber = int(instances.__len__() / 2)
    usedIndexes = []

    while teachingData.__len__() < instancesNumber:
        randomIndex = random.randint(0, instances.__len__() - 1)

        if randomIndex not in usedIndexes:
            teachingData.append(instances[randomIndex])
            usedIndexes.append(randomIndex)

    for i in range(instances.__len__()):
        if i not in usedIndexes:
            testData.append(instances[i])

    return teachingData, testData

def kNNAlgorithm(k, teachingData, testData, features, metric):
    teachingDataSet, teachingDataLabels = prepareData(teachingData, features)
    testDataSet, testDataLabels = prepareData(testData, features)

    classifier = KNeighborsClassifier(n_neighbors=k, metric=metric)
    classifier.fit(teachingDataSet, teachingDataLabels)

    predictions = classifier.predict(testDataSet)
    score = eval('accuracy_score')(testDataLabels, predictions)

    return score


def prepareData(data, features):
    finalData = []
    finalDataLabels = []

    for instance in data:
        featureSet = []
        finalDataLabels.append(instance.getCancerClass())

        for feature in features:
            featureSet.append(float(instance.getFeature(feature)))

        finalData.append(featureSet)

    return finalData, finalDataLabels

def crossValidation(kValues, metrics, instances, features):
    scores = {}

    for m in metrics:
        scores[m] = {}

        for k in kValues:
            scores[m][k] = []

            for i in range(0, features.__len__()):
                featuresIds = [feature.getParamID() for feature in features[0:i + 1]]
                data, dataLabels = prepareData(instances, featuresIds)

                knnClassifier = KNeighborsClassifier(n_neighbors = k, metric = m)
                rkf = RepeatedKFold(n_splits = 2, n_repeats = 5)
                score = cross_val_score(estimator = knnClassifier, X = data, y = dataLabels, scoring = 'accuracy', cv = rkf)
                scores[m][k].append(mean(score))

    return scores


main()
