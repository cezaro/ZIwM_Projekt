import pandas as pd

data = pd.read_csv("Resources/wdbc.csv", sep=',', header=None).transpose().values

kNN = [1, 5, 10]
metricType = ['mangattan', 'euclidean', 'minkowski']

values = []


# data[2][1] --> 3cia kolumna 2gi wiersz
print(data[2][1])

