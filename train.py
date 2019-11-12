import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def learningRateFromError(historytetas, learning_rate):
    teta0 = historytetas[0]
    teta1 = historytetas[1]
    if (len(teta0) == 1):
        return learning_rate
    delta_teta0 = teta0[len(teta0) - 1] - teta0[len(teta0) - 2]
    delta_teta1 = teta1[len(teta1) - 1] - teta1[len(teta1) - 2]
    gain0 = abs(delta_teta0 / teta0[len(teta0) - 1])
    gain1 = abs(delta_teta1 / teta1[len(teta1) - 1])
    if gain0 < 0 or gain1 < 0:
        return learning_rate * 0.5
    else:
        return learning_rate * 1.2




def gradientDescent(mileage_data, price_data, teta0, teta1, historytetas, learning_rate):
    # for i in range(50000):
    tmp_teta0 = 0
    tmp_teta1 = 0
    for mil, price in zip(mileage_data, price_data):
        x = mil
        y = price
        tmp_teta0 += teta0 + (teta1 * x) - y
        tmp_teta1 += (teta0 + (teta1 * x) - y) * x
        print(tmp_teta0, tmp_teta1)
    
    tmp_teta0 = (tmp_teta0 * learning_rate) / len(mileage_data)
    tmp_teta1 = (tmp_teta1 * learning_rate) / len(mileage_data)
    historytetas[0].append(tmp_teta0)
    historytetas[1].append(tmp_teta1)
    new_learning_rate = learningRateFromError(historytetas, learning_rate)
    print('apres', tmp_teta0, tmp_teta1)

    return [(teta0 - tmp_teta0, teta1 - tmp_teta1), new_learning_rate]

def	normalizeData(mileages, prices):
	x = []
	y = []
	minM = min(mileages)
	maxM = max(mileages)
	for mileage in mileages:
		x.append((mileage - minM) / (maxM - minM))
	minP = min(prices)
	maxP = max(prices)
	for price in prices:
		y.append((price - minP) / (maxP - minP))
	return (x, y)

if __name__=='__main__':
    data = pd.read_csv('data.csv')
    # print(data)
    iteration = 5
    learning_rate = 0.5
    mileage_data = data.loc[:, 'km']
    price_data = data.loc[:, 'price']
    mileage_data, price_data = normalizeData(mileage_data, price_data)
    # historyteta0 = []
    # historyteta1 = []
    lg = DataLR(learning_rate, mileage_data, price_data)
    historytetas = (lg.historyteta0, lg.historyteta1)
    # print(mileage_data, '\n\n ', price_data)
    # print(type(mileage_data), price_data)
    learning_rate = 0.1
    tetas = (0, 0)
    fig1 = plt.figure()
    plt.plot([x for x in mileage_data], [y for y in price_data], 'ro')
    t1 = np.array([0.0, 1.0])
    for i in range(iteration):
        tetas, learning_rate = gradientDescent(mileage_data, price_data, tetas[0], tetas[1], historytetas, learning_rate)
        print ('learning_rate:', learning_rate)
    plt.plot(t1, tetas[1] * t1 + tetas[0])
    fig2 = plt.figure()
    t2 = np.linspace(0.0, iteration, iteration)
    plt.plot(t2, historytetas[0])
    plt.plot(t2, historytetas[1])
    plt.show()
    print(tetas)
    # print(tetas[1] * 125000 + tetas[0])
    
