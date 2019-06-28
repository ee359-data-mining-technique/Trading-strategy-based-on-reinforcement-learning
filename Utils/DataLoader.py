'''
This is the Data Loader file:
    1. provide shuffled data
    2. provide sequence data
    3. provide get_next
'''
#-*- coding:utf-8 -*-
import os
import pandas as pd
import numpy as np
import pickle                  # OPTIONAL, if not needed, ignore it
import re

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

class DataLoader:
    def __init__(
            self, 
            seq_len = 60, # observation length
            jump = 3, # price change record length
            shuffle = True
    ):
        self.seq_len = seq_len
        self.jump = jump
        self.shuffle = shuffle

        self.train_observations = None
        self.train_price_changes = None
        
        self.test_observations = None
        self.test_price_changes = None

        # Batch Count
        self.train_batch_count = 0
        self.test_batch_count = 0

        # # Data Columns
        # self.data_cols = COLS
    
    def load_dataset(self):
        DataFrame = pd.read_csv(DATAROOT)
        
        features = DataFrame[FEATURE_COLS].values
        prices = DataFrame[COLS].values

        # check the data is morning or afternoon: using re-match
        time_stamp = DataFrame["UpdateTime"].values
        r = re.compile('.[901].*')
        vmatch = np.vectorize(lambda x:bool(r.match(x)))
        morning_selection = vmatch(time_stamp)

        observations = []
        price_changes = []
        
        for i in range(self.seq_len, len(features), self.jump):
            if sum(morning_selection[i-self.seq_len:i]) != self.seq_len and \
               sum(morning_selection[i-self.seq_len:i]) != 0:
                continue # check whether it is of the same half-day 
            observations.append(features[i-self.seq_len:i])
            price_changes.append(
                prices[i] - prices[i-self.jump]
            )

        observations = np.array(observations)
        price_changes = np.array(price_changes)

        self.train_observations = observations[:int(len(observations)*0.6)]
        self.train_price_changes = price_changes[:int(len(observations)*0.6)]
        self.train_length = len(self.train_observations)
        
        self.test_observations = observations[int(len(observations)*0.6)+1:]
        self.test_price_changes = price_changes[int(len(observations)*0.6)+1:]
        self.test_length = len(self.test_observations)

        print("Load dataset complete")

    def load_dataset_pickle(self):
        fr_train_data = open("./data/train_observations.pkl", 'rb')
        fr_train_label = open("./data/train_price_changes.pkl", 'rb')
        fr_test_data = open("./data/test_observations.pkl", 'rb')
        fr_test_label = open("./data/test_price_changes.pkl", 'rb')

        self.train_observations = pickle.load(fr_train_data)
        self.train_price_changes = pickle.load(fr_train_label)
        self.test_observations = pickle.load(fr_test_data)
        self.test_price_changes = pickle.load(fr_test_label)
        
        fr_train_data.close()
        fr_train_label.close()
        fr_test_data.close()
        fr_test_label.close()

        self.train_length = len(self.train_observations)
        self.test_length = len(self.test_observations)

        print("Load pickle dataset complete")

    def save_dataset_pickle(self):
        self.save_to_pickle(self.train_observations, "train_observations")
        self.save_to_pickle(self.train_price_changes, "train_price_changes")
        self.save_to_pickle(self.test_observations, "test_observations")
        self.save_to_pickle(self.test_price_changes, "test_price_changes")
        
    def save_to_pickle(self, data, name):
        fw = open(name + ".pkl", 'wb')
        pickle.dump(data, fw)
        fw.close()

    def next_batch_train(self, batch_num):
        if self.train_batch_count + batch_num > self.train_length:
            batch_obervations = self.train_observations[self.train_batch_count:]
            batch_price_changes = self.train_price_changes[self.train_batch_count:]            
            
            self.train_batch_count = 0
            return batch_obervations, batch_price_changes
        batch_obervations = self.train_observations[self.train_batch_count:self.train_batch_count+batch_num]
        batch_price_changes = self.train_price_changes[self.train_batch_count:self.train_batch_count+batch_num]
        self.train_batch_count = self.train_batch_count + batch_num
        return batch_obervations, batch_price_changes

if __name__ == "__main__":
    Data = DataLoader()
    Data.load_dataset()
    Data.save_dataset_pickle()
    Data.load_dataset_pickle()