from ENV.StockEnv import StockEnv
from Models.DeepQNetwork import DeepQNetwork
from Models.StupidAgent import StupidAgent
from Models.ActorCritic import DDPG
import numpy as np

env = StockEnv()
RL = DeepQNetwork(n_actions = 3, n_features = 108*60)
# RL = StupidAgent()
# RL = DDPG()

step = 0
for episode in range(300):
    # initial observation
    observation = env.reset()

    actions = []
    buy_prices = []
    sell_prices = []
    while True:
        # RL choose action based on observation
        action = RL.choose_action(observation)
        actions.append(action)

        # RL take action and get next observation and reward
        observation_, reward, done, price = env.step(action)
        if action == 0:
            buy_prices.append(price)
        if action == 2:
            sell_prices.append(price)

        RL.store_transition(observation, action, reward, observation_)

        if (step > 200) and (step % 5 == 0):
            RL.learn()

        # swap observation
        observation = observation_

        # break while loop when end of this episode
        if done:
            break
        step += 1

    actions = np.array(actions)
    buy_prices = np.array(buy_prices)
    sell_prices = np.array(sell_prices)
    print("Number of Buys: %d\t Number of Sells: %d\t Number of Idles: %d" %(
        sum(actions == 0), sum(actions == 2), sum(actions == 1)
    ))
    print("Average buying price: %f\t Average selling price: %f" %(
        np.mean(buy_prices), np.mean(sell_prices)    
    ))

# RL.plot_cost()