import pandas as pd
import numpy as np

import pickle

# def save_to_pickle(data, name):
#     fw = open(name + ".pkl", 'wb')
#     pickle.dump(data, fw)
#     fw.close()

# DATAROOT = "./Data/data.csv"
# jump = 30 # take price every 10s

# DataFrame = pd.read_csv(DATAROOT)
# prices = DataFrame["midPrice"].values

# prices = np.array([prices[i] for i in range(0, len(prices), jump)])

# save_to_pickle(prices, "prices")

fr = open("prices.pkl", 'rb')
p = pickle.load(fr)
fr.close()