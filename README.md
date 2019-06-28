# Trading strategy based on reinforcement learning

### Task Introduction

This task involves reinforcement learning and will no longer use the tags in task 1. To simplify the problem,
this task sets that:

- each tick has at most 5 hand long positions and 5 hand short positions. Long positions and short positions cannot be held at the same time. 
- A tick can only have one action at a time. Positions can be increased or decreased (with unit equals one hand) through buying and selling, and the absolute value of change in the number of positions of one action cannot exceed one hand. The current state can be maintained by an idle action.

- When the buying action is executed, the purchase will be successful and will not have any impact on the market. The price is AskPrice1 of the current tick. When the selling action is executed, the sell will be successful and will have no effect on the market. The price is BidPrice1 of the current tick. 

Finally, you should include in your report: **the number of buying and spelling on testing set**, **the average price to buy and the average price to sell.** Besides, attach **action selection for each tick on testing set for submission**.

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