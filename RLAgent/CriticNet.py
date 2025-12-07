import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / (fan_in ** 0.5)
    return (-lim, lim)

class Critic(nn.Module):
    def __init__(self, state_size, action_size, max_action=1.0, fcs1_units=256, fc2_units=128):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, fcs1_units)
        self.ln1 = nn.LayerNorm(fcs1_units)
        
        self.fc2 = nn.Linear(fcs1_units + action_size, fc2_units)
        self.ln2 = nn.LayerNorm(fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        
        self.max_action = max_action

        self.reset_parameters()
        
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, state, action):
        xs = self.fc1(state)
        xs = self.ln1(xs)
        xs = F.relu(xs)
        
        x = torch.cat((xs, action), dim=1)
        
        x = self.fc2(x)
        x = self.ln2(x)
        x = F.relu(x)
        
        q_value = self.fc3(x)
        return q_value