from scipy.stats import ks_2samp
from ksType import ksType
from instance import Instance


def main():
    fileName = 'Resources/breast-cancer-wisconsin.data'
    # kNN = [1, 5, 10]
    # metricType = ['mangattan', 'euclidean', 'minkowski']
    quantityOfFeatures = 9
    instances = loadDataFromFile(fileName)
    ranking = kolmogorovTest(instances, quantityOfFeatures)

    for feature in range(0, ranking.__len__()):
        print(ranking[feature].getParamID(), '\t', ranking[feature].getPValue(), '\t', ranking[feature].getStatistic())


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


main()
