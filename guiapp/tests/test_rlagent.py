import unittest
import torch
import numpy as np
from RLAgent.RLAgent import RLAgent

class TestRLAgent(unittest.TestCase):
    def test_rl_agent_initialization_and_action(self):
        # Define parameters for the RLAgent
        state_size = 4
        action_size = 2
        max_action = 1.0
        device = torch.device("cpu")

        # Initialize the RLAgent
        agent = RLAgent(state_size, action_size, max_action, device)

        # Check if the agent's components are initialized correctly
        self.assertIsNotNone(agent.actor_local)
        self.assertIsNotNone(agent.critic_local)

        # Create a dummy state
        state = np.random.rand(state_size)

        # Choose an action
        action = agent.choose_action(state)

        # Check if the action is within the specified range
        self.assertTrue(np.all(action >= -max_action))
        self.assertTrue(np.all(action <= max_action))

if __name__ == '__main__':
    unittest.main()