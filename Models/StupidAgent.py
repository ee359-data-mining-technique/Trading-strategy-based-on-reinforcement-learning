import random
import numpy as np

class StupidAgent:
    def __init__(self):
        self.position = random.randint(0,5)
    
    def choose_action(self, observation):
        if self.position == 0:
            action = np.random.randint(0, 2)
        elif self.position == 5:
            action = np.random.randint(1, 3)
        else:
            action = np.random.randint(0, 3)
        if action == 0:
            self.position += 1
        if action == 2:
            self.position -= 1
        return action

    def store_transition(self, s, a, r, s_):
        pass

    def learn(self):
        pass