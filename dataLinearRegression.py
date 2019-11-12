import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from train import normalizeData 

class DataLR:
    def __init__(self, base_learningRate, dataX, dataY, teta0=0.0, teta1=0.0):
        self.teta0 = teta0
        self.teta1 = teta1
        self.gain_teta0 = 0.0
        self.gain_teta1 = 0.0
        self.historyGain0 = [self.gain_teta0]
        self.historyGain1 = [self.gain_teta1]
        self.historyTeta0 = []
        self.historyTeta1 = []
        self.learning_rate = base_learningRate
        self.dataX = dataX
        self.dataY = dataY
        self.iteration_tot = 0
        self.lenData = len(dataX)

    def getTetas(self):
        return (self.teta0, self.teta1)

    def getDeltaTetas(self):
        tmp_teta0 = 0
        tmp_teta1 = 0
        for mil, price in zip(self.dataX, self.dataY):
            x = mil
            y = price
            tmp_teta0 += self.teta0 + (self.teta1 * x) - y
            tmp_teta1 += (self.teta0 + (self.teta1 * x) - y) * x
            # print(tmp_teta0, tmp_teta1)
        tmp_teta0 = (tmp_teta0 * self.learning_rate) / self.lenData
        tmp_teta1 = (tmp_teta1 * self.learning_rate) / self.lenData
        return (tmp_teta0, tmp_teta1)

    def upddateTetas(self, tmp_teta0, tmp_teta1):
        self.teta0 -= tmp_teta0
        self.teta1 -= tmp_teta1

    def updateHistoryTetas(self, teta0, teta1):
        self.historyTeta0.append(teta0)
        self.historyTeta1.append(teta1)

    def getGains(self, old_Dteta0, new_Dteta0, old_Dteta1, new_Dteta1):
        gain0 = old_Dteta0 - new_Dteta0
        gain1 = old_Dteta1 - new_Dteta1
        return (gain0, gain1)

    def updateHistoryGains(self, gain0, gain1):
        self.historyGain0.append(gain0)
        self.historyGain1.append(gain1)


    def improveLearningRate_andIterate(self, oldGain0, gain0, oldGain1, gain1):
        gradGain0 = abs(gain0 / oldGain0)
        gradGain1 = abs(gain1 / oldGain1)
        if gradGain0 < 0.01 and gradGain1 < 0.01:
            return False
        if gradGain0 > 0.7 and gradGain1 > 0.7:
            self.learning_rate *= 1.2
        else:
            self.learning_rate *= 0.5
        return True

    def upIteration(self):
        self.iteration_tot += 1

if __name__=='__main__':
    data = pd.read_csv('data.csv')
    iteration = 80
    learning_rate = 0.5
    mileage_data = data.loc[:, 'km']
    price_data = data.loc[:, 'price']
    mileage_data, price_data = normalizeData(mileage_data, price_data)
    lg = DataLR(learning_rate, mileage_data, price_data)
    dTeta0 = 0.0
    dTeta1 = 0.0
    i = 0
    while i < iteration:
        dTeta0, dTeta1 = lg.getDeltaTetas()
        lg.updateHistoryTetas(dTeta0, dTeta1)
        lg.upddateTetas(dTeta0, dTeta1)
        i += 1

    teta0, teta1 = lg.getTetas()
    fig1 = plt.figure()
    plt.plot([x for x in mileage_data], [y for y in price_data], 'ro')
    t1 = np.array([0.0, 1.0])
    plt.plot(t1, teta1 * t1 + teta0)
    fig2 = plt.figure()
    t2 = np.linspace(0.0, iteration, iteration)
    plt.plot(t2, lg.historyTeta0, label='teta0')
    plt.plot(t2, lg.historyTeta1, label='teta1')
    plt.show()


