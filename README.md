# Trading strategy based on reinforcement learning

### Task Introduction

This task involves reinforcement learning and will no longer use the tags in task 1. To simplify the problem,
this task sets that:

- each tick has at most 5 hand long positions and 5 hand short positions. Long positions and short positions cannot be held at the same time. 
- A tick can only have one action at a time. Positions can be increased or decreased (with unit equals one hand) through buying and selling, and the absolute value of change in the number of positions of one action cannot exceed one hand. The current state can be maintained by an idle action.

- When the buying action is executed, the purchase will be successful and will not have any impact on the market. The price is AskPrice1 of the current tick. When the selling action is executed, the sell will be successful and will have no effect on the market. The price is BidPrice1 of the current tick. 

Finally, you should include in your report: **the number of buying and spelling on testing set**, **the average price to buy and the average price to sell.** Besides, attach **action selection for each tick on testing set for submission**.

### Necessary Dependencies

We use python = 3.5+ for this project, you should prepare following packages:

- tensorflow 
- numpy
- pandas
- pickle

### Data Preparation

To run this code, you should download corresponding dataset: **data.csv** and put it in the directory *./Data/data.csv*. The directory of this whole project should be like:

```python
-- Data
---- data.csv
-- Models
-- Utils
-- ENV
-- run.py
```

### How to Run

To see how to run, you should refer to *./run.py*:

```python
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
```

### More precise Settings

- We have 5 positions for buying and selling. Thus, we can set the position to [0, 5]. When position = 0, we can only buy stock or idle. When position = 5, we can only sell stock or idle. This should be restrained in models.

- At each time tick, we can only buy or sell one hand of stock. Thus the position either +1 or -1.
- **(Set Reward to Guide Policy)** We denote BidPrice1 at time $t$ as $bp_t$ and AskPrice1 at time $t$ as $ap_t$. $\delta_t$ as decision made at time $t$. $\delta_t \in \{+1, 0, -1\}$ as for buy, idle, and sell. If we buy one hand stock at time $t$ ($\delta_t = +1$), the reward agent gets is $(bp_t - bp_{t-1}) \delta_t$. If we sell one hand stock at time $t$ ($\delta_t = -1$), the reward agent gets is $(ap_t - ap_{t-1})\delta_t$. Set $z_t^1 = bp_t - bp_{t-1}$ and $z_t^2 = ap_t - ap_{t-1}$. Input feature is $f_t = [z_{t-m+1}^1, \ldots, z_{t}^1, z_{t-m+1}^2, \ldots, z_t^2]$ 

### Data for reinforcement learning

- training feature: 

  - $f_t = [z_{t-m+1}^1, \ldots, z_{t}^1, z_{t-m+1}^2, \ldots, z_t^2]$ 

  - extracted features

- training label/ training reward: 

  - $(bp_t - bp_{t-1}) \delta_t$ if $\delta_t = +1$ 
  - $(ap_t - ap_{t-1})\delta_t$ if $\delta_t = -1 $

  - $ + \ c |\delta_t - \delta_{t-1}|$