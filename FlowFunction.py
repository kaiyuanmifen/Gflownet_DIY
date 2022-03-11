
import torch
import numpy as np

import torch.nn as nn

import torch.nn.functional as F

class FlowFunction(nn.Module):
 
    def __init__(self, state_dim, n_embed):
        super().__init__()

        self.embedding_dim = 8
        self.hidden_dim=32
        self.state_dim = state_dim
        self.n_embed=n_embed
        
        self.embed = nn.Embedding(n_embed, self.embedding_dim)

        self.FFL1= nn.Linear(self.embedding_dim+state_dim, self.hidden_dim)
        self.FFL2= nn.Linear(self.hidden_dim, self.hidden_dim)
        self.outputlayer= nn.Linear(self.hidden_dim, 1)


    def forward(self, state,action):       
        '''
        state has shape (bsz,state_dim)
        action is a one hot vector has shape (bsz,n_action_space)
        This function output Log(flow) for numeric issues
        '''

        emebeded_actions=self.embed_code(torch.argmax(action,1))
        

        x=torch.cat([state,emebeded_actions],1)
        


        x=nn.LeakyReLU()(self.FFL1(x))
        x=nn.LeakyReLU()(self.FFL2(x))
        output=self.outputlayer(x)


        return output


    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.weight)


if __name__ == "__main__":
    Fnet=FlowFunction(state_dim=3, n_embed=8)

    states=torch.zeros((2,3))
    actions=torch.tensor([[0,1,0,0,0,0],[0,0,0,0,1,0]])

    output=Fnet.forward(states,actions)

    print("output")
    print(output)