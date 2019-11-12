import numpy as np
import csv

def get_data(m_list, p_list):
    mil_list = []
    price_list = []
    matrix = []
    with open("data.csv") as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            matrix.append(row[0])
            matrix.append(row[1])
    matrix.pop(0)
    matrix.pop(0)
    #X = matrix[:, 1]
    m = []
    while matrix != []:
        m.append(matrix[:2])
        matrix = matrix[2:]

    #print(m)
    arr = np.array(m)
    Xmin = min(arr[:, 1])
    Xmax = max(arr[:, 1])
    Xs = arr[:, 1]
    print(Xs)
    farr = arr.astype(float)
    print(arr)
    print(farr)
    print("min = " + Xmin)
    print("max = " + Xmax)
    farr[:, 1] = (Xs.astype(float) - float(Xmin)) / (float(Xmax) - float(Xmin))
    print(farr)

if __name__ == '__main__':
    m_list = []
    p_list = []
    get_data(m_list, p_list)
    #print(m_list)
    #print(p_list)
