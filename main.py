import pandas as pd

fileName = 'Resources/wdbc.data'

def main():
    file = open(fileName, 'r')
    lines = file.readlines() 
  

    # Array with data
    data = []

    # Reading data from file
    for line in lines: 
        data.append(line.strip().split(','))


    # data = pd.read_csv(fileName, sep=',', header = None).transpose().values

    kNN = [1, 5, 10]
    metricType = ['mangattan', 'euclidean', 'minkowski']

    values = []


    # data[2][1] --> 3cia kolumna 2gi wiersz
    # print(data)


main()