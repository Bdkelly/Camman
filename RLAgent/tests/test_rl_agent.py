import pytest
import torch
import numpy as np
from RLAgent.RLAgent import RLAgent

@pytest.fixture
def agent():
    return RLAgent(state_size=3, action_size=1, max_action=1.0, device='cpu')

def test_choose_action(agent):
    state = np.array([0.1, 0.2, 0.3])
    action = agent.choose_action(state)
    assert isinstance(action, np.ndarray)
    assert action.shape == (1,)
    assert -1.0 <= action[0] <= 1.0

def test_add_experience(agent):
    state = np.array([0.1, 0.2, 0.3])
    action = np.array([0.5])
    reward = 1.0
    next_state = np.array([0.4, 0.5, 0.6])
    done = False
    agent.add_experience(state, action, reward, next_state, done)
    assert len(agent.memory) == 1

def test_learn(agent):
    # Add enough experience to trigger learning
    for _ in range(agent.batch_size):
        state = np.random.rand(3)
        action = np.random.rand(1)
        reward = np.random.rand()
        next_state = np.random.rand(3)
        done = False
        agent.add_experience(state, action, reward, next_state, done)
    
    # Check that actor and critic losses are updated
    initial_actor_loss = agent.actor_loss
    initial_critic_loss = agent.critic_loss
    agent.learn()
    assert agent.actor_loss != initial_actor_loss
    assert agent.critic_loss != initial_critic_loss
