from re import M
import numpy as np
import torch
import torch.nn as nn
from alg_parameters import *
from transformer.encoder_model import TransformerEncoder
import torch.nn.functional as F

class Communication(nn.Module):
    
    def __init__(self):
        # super().__init__()
        super(Communication, self).__init__()
        # self.num_agent = EnvParameters.N_AGENTS
        self.pred_t_lmt = TrainingParameters.PREDICTION_TIMESTEPS_LIMIT
        ''' In following transformers, we have (Batch, 10, 512) as input.
        And (Batch, 10, 1024) as output. '''

        # If this doesn't works. then do all with single transformers

       
        self.t_intra = TransformerEncoder(d_model=NetParameters.D_MODEL+2*(TrainingParameters.PREDICTION_TIMESTEPS_LIMIT+1)+ TrainingParameters.PREDICTION_TIMESTEPS_LIMIT//2,
                                                      d_hidden=NetParameters.D_HIDDEN//2,
                                                      n_layers=NetParameters.N_LAYERS, n_head=NetParameters.N_HEAD,
                                                      d_k=NetParameters.D_K,
                                                      d_v=NetParameters.D_V, n_position=NetParameters.N_POSITION//2)
        
        self.linear2 = nn.Linear(TrainingParameters.PREDICTION_TIMESTEPS_LIMIT, TrainingParameters.PREDICTION_TIMESTEPS_LIMIT//2)
        self.linear3 = nn.Linear(2*(TrainingParameters.PREDICTION_TIMESTEPS_LIMIT+1), 2*(TrainingParameters.PREDICTION_TIMESTEPS_LIMIT+1))

    def forward(self, message, obs, comms_mask_groupwise, predicted_actions, curr_pos_time, device):
        '''Here, my algorithm will be formed'''

        self.comms_mask = comms_mask_groupwise  
        '''
            message = 1,8, NET_SIZE
            obs = 1,4,FOV_SIZE, FOV_SIZE
            predicted_action = 1,N_AGENTS,PREDICTION_TIMESTEP_LIMIT
        '''
        predicted_actions = predicted_actions.to(device)
        predicted_actions = F.relu(self.linear2(predicted_actions))
        curr_pos_time = curr_pos_time.to(device)
        curr_pos_time = F.relu(self.linear3(curr_pos_time))
        # print(message.shape, curr_pos_time.shape, predicted_actions.shape)
        self.message = torch.cat([message, curr_pos_time, predicted_actions], 2)

        t_intra, attns = self.t_intra(self.message, self.comms_mask)
         
        return t_intra.to(device), attns

