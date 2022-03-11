
import torch
import numpy as np

class MultiModalSeq(object):
    def __init__(self, vector_Len,N_values,batch_size=32):
        '''
        zeros are "empty states", values start from 1
        '''

        self.vector_Len = vector_Len
        self.N_values = N_values
        self.batch_size=batch_size
        self.states=torch.zeros((batch_size,vector_Len))

        self.trajectories=[] # In this simplfied env. all trajectories have the same length
        self.trajectories.append(self.states)

        self.action_indice=torch.arange(0,vector_Len*N_values).reshape(vector_Len,N_values)
        self.action_space=torch.nn.functional.one_hot(self.action_indice.flatten())

        self.rewards=torch.zeros(self.batch_size)

    def reset(self):
        self.states=torch.zeros((self.batch_size,self.vector_Len))
        
        self.trajectories=[]
        self.states

        self.rewards=torch.zeros(self.batch_size)

    def UpdateState(self,state,action):
        action_index=torch.argmax(action)

        value_to_add=action_index%self.N_values+1

        index_position=action_index//self.N_values


        state[index_position]=value_to_add

     


        return state

    def get_flow_in(self,state,FlowFunction):
        ####get the in flow of a state, in this env all state transition are deterministic
        
        #####calculate the backward actions from the state
        backward_actions=[]
        parent_states=[]

        for index_position in range(state.shape[0]):
            if state[index_position]!=0:
                backward_action=state[index_position]
                backward_action_index=index_position*self.N_values+backward_action-1
                backward_action_index=backward_action_index.long()

                vec_state=state.detach().clone()
                vec_state[index_position]=vec_state[index_position]-backward_action
                
                parent_states.append(vec_state.unsqueeze(0))
                backward_actions.append(self.action_space.clone().detach()[backward_action_index,:].unsqueeze(0))

        parent_states=torch.cat(parent_states,0)
        backward_actions=torch.cat(backward_actions,0)

        LogFlows=FlowFunction(parent_states,backward_actions)

        return parent_states,backward_actions,LogFlows


    def get_flow_out(self,state,FlowFunction):
        ####get the out flow of a state
        terminal=(state>0).sum()==self.vector_Len   
        
   
        #make sure it is acyclic    
        action_indice_allowed=self.action_indice.clone().detach()

        action_indice_allowed=action_indice_allowed[state==0,:]
        
        action_indice_allowed=action_indice_allowed.flatten()

        PossibleActions=self.action_space[action_indice_allowed,:]

        state_vec=state.unsqueeze(0).repeat(action_indice_allowed.shape[0],1)

        LogFlows=FlowFunction(state_vec,PossibleActions)
   
     
        return state,PossibleActions,LogFlows

    def get_reward(self,state):

        terminal=(state>0).sum()==self.vector_Len    
        
        if terminal:
            reward=((state[0]-state[2])**2).item()+0.01 #make a minimum reward of 0.01
        else: 
            reward=0

        return reward


    def step_forward(self, FlowFunction):


        terminal=(self.states>0).sum(1)==self.vector_Len      

        for i in range(self.batch_size):
            state=self.states.clone().detach()[i,]

            if not terminal[i]:
                    
                _,PossibleActions,LogFlows=self.get_flow_out(state,FlowFunction)
                Flows=torch.exp(LogFlows)
                action_Prob=torch.nn.Softmax(0)(Flows)

                action_chosen=np.random.choice(np.arange(0, PossibleActions.shape[0]), p=action_Prob.flatten().detach().numpy())
                
                action=PossibleActions[action_chosen,:]

                state=self.UpdateState(state,action)

                
                self.states[i,]=state

            self.rewards[i]=self.get_reward(state)

        self.trajectories.append(self.states.detach().clone())
        
        return self.states, self.rewards


    def CalculateFlowMatchingLoss(self, FlowFunction):
        '''
        first run the step_forward function to generate batch of trajectgories ending with terminal states
        then run this function to calculate flow matching loss 
        '''

        ####low match on all but initial state
        Epsilon=0.01 #small number to avoid numeric problem to avoid log on tiny number, and balance large and small flow

        Match_loss_all=0
        for t in range(1,len(self.trajectories)):
            states_t=self.trajectories[t] #shape Batch_size X state_dim

            for i in range(states_t.shape[0]):
                

                state=states_t[i,:]

                terminal=(state>0).sum()==self.vector_Len   
           
                if not terminal:
                    _,_,LogFlowsOut=self.get_flow_out(state,FlowFunction)
                else:
                    LogFlowsOut=torch.tensor(0.0)

                _,_,LogFlowsIn=self.get_flow_in(state,FlowFunction)
                
                Reward=self.get_reward(state)

                In_log_sum=torch.log(Epsilon+torch.exp(LogFlowsIn).sum())
                Out_log_sum=torch.log(Epsilon+Reward+torch.exp(LogFlowsOut).sum())
                
           

                Match_loss_ti=(In_log_sum-Out_log_sum)**2
 
                Match_loss_all+=Match_loss_ti

        return Match_loss_all

          



    
    