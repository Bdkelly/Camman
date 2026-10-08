import numpy as np
import pytest

from training.reinforcement.agent import RLAgent


@pytest.fixture
def agent():
    return RLAgent(state_size=3, action_size=1, max_action=1.0, device="cpu")


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


def test_updates_networks_without_target_or_critic_actor_gradients():
    import torch

    agent = RLAgent(3, 1, 1, "cpu", batch_size=8, memory_size=16)
    rng = np.random.default_rng(5)
    for _ in range(8):
        agent.add_experience(
            rng.normal(size=3), rng.uniform(-1, 1, 1), 1, rng.normal(size=3), False
        )
    actor_before = [p.detach().clone() for p in agent.actor_local.parameters()]
    critic_before = [p.detach().clone() for p in agent.critic_local.parameters()]
    agent.learn()
    assert any(not torch.equal(a, b) for a, b in zip(actor_before, agent.actor_local.parameters()))
    assert any(
        not torch.equal(a, b) for a, b in zip(critic_before, agent.critic_local.parameters())
    )
    assert all(p.grad is None for p in agent.actor_target.parameters())
    assert all(p.grad is None for p in agent.critic_target.parameters())
    assert all(p.grad is None and p.requires_grad for p in agent.critic_local.parameters())
    assert agent.updates == 1


def test_replay_wrap_and_restore_retains_numeric_samples():
    import random

    import torch

    from training.reinforcement.agent import ReplayBuffer, Transition

    replay = ReplayBuffer(3, 2, 1)
    for index in range(5):
        replay.append(
            Transition(
                torch.full((2,), float(index)),
                torch.tensor([0.1]),
                torch.tensor([1.0]),
                torch.ones(2),
                torch.zeros(1),
            )
        )
    restored = ReplayBuffer(3, 2, 1)
    restored.load_state_dict(replay.state_dict())
    assert len(restored) == 3 and restored.cursor == 2
    first, second = replay.sample(random.Random(1), 3), restored.sample(random.Random(1), 3)
    for a, b in zip(first, second):
        torch.testing.assert_close(a, b)
    assert sorted(first.state[:, 0].tolist()) == [2, 3, 4]
