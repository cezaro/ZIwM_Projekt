from scipy.stats import ks_2samp
from ksType import ksType
from instance import Instance
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import random


def main():
    fileName = 'Resources/breast-cancer-wisconsin.data'
    # fileName = 'Resources/wdbc.data'

    # kNN = [1, 5, 10]
    # metricType = ['manhattan', 'euclidean', 'minkowski']

    quantityOfFeatures = 9
    instances = loadDataFromFile(fileName)
    ranking = kolmogorovTest(instances, quantityOfFeatures)
    featuresIds = [x.getParamID() for x in ranking]

    # for feature in range(0, ranking.__len__()):
    #     print(ranking[feature].getParamID() + 1, '\t', ranking[feature].getPValue(), '\t', ranking[feature].getStatistic())

    # 2-krotna walidacja krzyżowa
    teachingData, testData = crossValidation(instances)
    # print(instances.__len__(), teachingData.__len__(), testData.__len__())


    score = kNNAlgorithm(1, teachingData, testData, featuresIds, 'euclidean')
    print(score)


def loadDataFromFile(fileName):
    file = open(fileName, 'r').read()
    lines = file.split('\n')
    instances = []
    for line in lines:
        row = line.split(",")

        if (row.__len__() == 11):
            instance = Instance(row[0], row[1:10], row[10])
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

def crossValidation(instances):
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


main()
