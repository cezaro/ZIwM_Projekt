class Instance:
    id: int
    featureValues: list
    cancerType: int
    cancerClass: str

    def __init__(self, id, featureValues, cancerType):
        self.id = id
        self.featureValues = featureValues
        self.cancerType = cancerType

        if int(cancerType) == 2:
            self.cancerClass = 'B'
        else:
            self.cancerClass = 'M'

    def getId(self):
        return self.id

    def getFeatureValues(self):
        return self.featureValues

    def getFeature(self, feature):
        value = self.getFeatureValues()[feature]

        return value if value != '?' else 1

    def getCancerType(self):
        return self.cancerType

    def getCancerClass(self):
        return self.cancerClass