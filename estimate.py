import numpy as np
import csv

def estimatePrice(mileage):
    esPrice = teta0 + teta1 * mileage
    print('Mileage: %d | Estimated price %d' % (mileage, esPrice))

if __name__ == 'main':
    teta0 = 0
    teta1 = 0