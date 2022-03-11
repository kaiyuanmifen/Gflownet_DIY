
import torch
import numpy as np
import torch.optim as optim 
from MultiModalSeq import MultiModalSeq

from FlowFunction import FlowFunction

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('AGG')

N_values=4
vector_Len=3
N_epsiodes=2000

Env=MultiModalSeq(vector_Len=vector_Len,N_values=N_values,batch_size=32)

Fnet=FlowFunction(state_dim=vector_Len, n_embed=vector_Len*N_values)

Env.reset()

optimizer = optim.Adam(Fnet.parameters(), lr=1e-4)


Match_loss_all=[]
AllRewards=[]

for Episode in range(N_epsiodes):
    print("episode",Episode)
    Env.reset()
    #####forward, build the trajectory
    for step in range(vector_Len):
        states, rewards=Env.step_forward(Fnet)

        # print("states")
        # print(states)

        # print("rewards")
        # print(rewards)    
    #####calculate rewards
    AllRewards.append(rewards.mean().item())

    #####calculate flow match loss
    optimizer.zero_grad()

    Match_loss=Env.CalculateFlowMatchingLoss(Fnet)
    Match_loss_all.append(Match_loss.item())

    Match_loss.backward()

    optimizer.step()




plt.plot(list(range(N_epsiodes)), Match_loss_all)
plt.title("matching loss vs episode")
plt.xlabel("episode")
plt.ylabel("matching loss")
plt.savefig('matchinglossEpisode.png')

plt.clf()
    
plt.plot(list(range(N_epsiodes)), AllRewards)
plt.title("Rewards vs episode")
plt.xlabel("episode")
plt.ylabel("Rewards(mean)")
plt.savefig('RewardEpisode.png')
    