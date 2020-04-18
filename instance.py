class Instance:
    id: int
    featureValues: list
    cancerType: int

    def __init__(self, id, featureValues, cancerType):
        self.id = id
        self.featureValues = featureValues
        self.cancerType = cancerType

    def getId(self):
        return self.id

    def getFeatureValues(self):
        return self.featureValues

    def getCancerType(self):
        return self.cancerType