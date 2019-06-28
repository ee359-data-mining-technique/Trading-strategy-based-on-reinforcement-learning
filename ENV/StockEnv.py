'''
This file specify the stock price environment:
    reset():
        start from a random training day
    step(action):
        given an action, return a corresponding reward
        return next_observation, reward, done
'''
FEATURE_COLS = ["indicator1", "indicator2", "indicator3", "indicator4", "indicator5", "indicator6", 
        "indicator7", "indicator8", "indicator9", "indicator10", "indicator11", "indicator12",
        "indicator13", "indicator14", "indicator15", "indicator16", "indicator17", "indicator18",
        "indicator19", "indicator20", "indicator21", "indicator22", "indicator23", "indicator24",
        "indicator25", "indicator26", "indicator27", "indicator28", "indicator29", "indicator30", 
        "indicator31", "indicator32", "indicator33", "indicator34", "indicator35", "indicator36",
        "indicator37", "indicator38", "indicator39", "indicator40", "indicator41", "indicator42", 
        "indicator43", "indicator44", "indicator45", "indicator46", "indicator47", "indicator48",
        "indicator49", "indicator50", "indicator51", "indicator52", "indicator53", "indicator54",
        "indicator55", "indicator56", "indicator57", "indicator58", "indicator59", "indicator60",
        "indicator61", "indicator62", "indicator63", "indicator64", "indicator65", "indicator66", 
        "indicator67", "indicator68", "indicator69", "indicator70", "indicator71", "indicator72",
        "indicator73", "indicator74", "indicator75", "indicator76", "indicator77", "indicator78", 
        "indicator79", "indicator80", "indicator81", "indicator82", "indicator83", "indicator84",
        "indicator85", "indicator86", "indicator87", "indicator88", "indicator89", "indicator90",
        "indicator91", "indicator92", "indicator93", "indicator94", "indicator95", "indicator1",
        "indicator97", "indicator98", "indicator99", "indicator100", "indicator101", "indicator102", 
        "indicator103", "indicator104", "indicator105", "indicator106", "indicator107", "indicator108"]
COLS = ["AskPrice1", "BidPrice1"]
DATAROOT = "./Data/data.csv"

import pandas as pd
import numpy as np
import random
import re

seq_len = 60
jump = 30

class StockEnv:
    def __init__(self):
        DataFrame = pd.read_csv(DATAROOT)
        
        self.features = DataFrame[FEATURE_COLS].values
        self.prices = DataFrame[COLS].values
        self.length = len(self.features)

        # check the data is morning or afternoon: using re-match
        time_stamp = DataFrame["UpdateTime"].values
        r = re.compile('.[901].*')
        vmatch = np.vectorize(lambda x:bool(r.match(x)))
        self.morning_selection = vmatch(time_stamp)

        self.cur_idx = 0
        self.position = 0
        self.counter = 0

        self.actions = [+1, 0, -1] # corresponding actions for buy, idle, and sell

        print("Initilization Complete")

    def reset(self):
        self.cur_idx = random.randint(0, self.length-1)
        self.position = random.randint(0, 5)

        # if current index is in the morning, move to the first time of next afternoon
        if (sum(self.morning_selection[self.cur_idx: self.cur_idx+10]) == 10):
            while(sum(self.morning_selection[self.cur_idx: self.cur_idx+10]) != 0):
                self.cur_idx = (self.cur_idx+1)%self.length
        # if current index is in the afternoon, move to the first time of next morning
        elif (sum(self.morning_selection[self.cur_idx: self.cur_idx+10]) == 0):
            while(sum(self.morning_selection[self.cur_idx: self.cur_idx+10]) != 10):
                self.cur_idx = (self.cur_idx+1)%self.length
        # if current index is in the other area, move to next morning
        else:
            while(sum(self.morning_selection[self.cur_idx: self.cur_idx+10]) != 10):
                self.cur_idx = (self.cur_idx+1)%self.length
        
        print("Starting stock at %d" %(self.cur_idx))

        self.cur_idx += seq_len
        cur_observation = self.features[self.cur_idx - seq_len: self.cur_idx].flatten()

        self.counter = 0
        return cur_observation


    def step(self, action):
        done = False

        price = -1
        if self.actions[action] == +1:
            reward = (self.prices[self.cur_idx + jump] - self.prices[self.cur_idx])[1]
            price = self.prices[self.cur_idx][1]
        if self.actions[action] == -1:
            reward = ((self.prices[self.cur_idx + jump] - self.prices[self.cur_idx])[0])*-1
            price = self.prices[self.cur_idx][0]
        else:
            reward = 0

        self.cur_idx += jump
        cur_observation = self.features[self.cur_idx - seq_len: self.cur_idx].flatten()

        self.counter += 1
        if (self.counter > 6*60):
            done = True

        return cur_observation, reward, done, price
        

if __name__ == "__main__":
    env = StockEnv()
    env.reset()