from collections import deque

import torch
from torch import optim
from unityagents import UnityEnvironment
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from Agent import Agent, AgentDuo
from networks import ActorNetwork


def watch_agents_from_pth_file(env: UnityEnvironment, brain_name: str, agent_duo: AgentDuo, weight_folder_path: str):
    agent_duo.agent1.actor_local.load_state_dict(torch.load(f"{weight_folder_path}/checkpoint-actor1.pth"))
    agent_duo.agent1.critic_local.load_state_dict(torch.load(f"{weight_folder_path}/checkpoint-critic1.pth"))
    agent_duo.agent2.actor_local.load_state_dict(torch.load(f"{weight_folder_path}/checkpoint-actor2.pth"))
    agent_duo.agent2.critic_local.load_state_dict(torch.load(f"{weight_folder_path}/checkpoint-critic2.pth"))
    agent_duo.agent1.actor_local.eval()
    agent_duo.agent1.critic_local.eval()
    agent_duo.agent2.actor_local.eval()
    agent_duo.agent2.critic_local.eval()
    watch_agents(env, brain_name, agent_duo, episodes=10)


def train_agents(env: UnityEnvironment, brain_name: str, agent_duo: AgentDuo, n_episodes: int):
    """
    Trains the Agent Duo
    :param env:
    :param brain_name:
    :param agent_duo:
    :param n_episodes:
    :return:
    """
    total_scores: dict = {"agent1": [], "agent2": []}
    # store the last 100 scores into a queue to check if the agent reached the goal
    """ from project instructions:
    The task is episodic, [..] your agents must get an average score of +0.5 (over 100 consecutive episodes). Specifically,
    After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. 
    This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
    This yields a single score for each episode.
    """
    scores_window = deque(maxlen=100)

    for i_episode in range(1, n_episodes + 1):
        # reset the environment
        env_info = env.reset(train_mode=True)[brain_name]
        agent_duo.agent1.reset_noise()
        agent_duo.agent2.reset_noise()

        states = env_info.vector_observations
        # initialize the scores for the two agents
        scores = np.zeros(2)

        while True:
            action1 = agent_duo.agent1.act(states[0], add_noise=i_episode < 900)
            action2 = agent_duo.agent2.act(states[1], add_noise=i_episode < 900)
            # send both actions to the environment
            env_info = env.step([action1, action2])[brain_name]
            # get the next state (for each agent)
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            scores += rewards

            agent_duo.step(
                (states[0], action1, rewards[0], next_states[0], dones[0]),
                (states[1], action2, rewards[1], next_states[1], dones[1])
            )

            states = next_states
            if np.any(dones):
                break

        # episode has ended
        scores_window.append(np.max(scores))
        total_scores["agent1"].append(scores[0])
        total_scores["agent2"].append(scores[1])

        if i_episode % 100 == 0:
            print(f"""Episode {i_episode}: Average Score: {np.mean(scores_window):.2f}""")

        if np.mean(scores_window) >= 0.5:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode,
                                                                                         np.mean(scores_window)))
            torch.save(agent_duo.agent1.actor_local.state_dict(), 'weights/checkpoint-actor1.pth')
            torch.save(agent_duo.agent1.critic_local.state_dict(), 'weights/checkpoint-critic1.pth')
            torch.save(agent_duo.agent2.actor_local.state_dict(), 'weights/checkpoint-actor2.pth')
            torch.save(agent_duo.agent2.critic_local.state_dict(), 'weights/checkpoint-critic2.pth')
            break
    torch.save(agent_duo.agent1.actor_local.state_dict(), 'weights/loosing-checkpoint-actor1.pth')
    torch.save(agent_duo.agent1.critic_local.state_dict(), 'weights/loosing-checkpoint-critic1.pth')
    torch.save(agent_duo.agent2.actor_local.state_dict(), 'weights/loosing-checkpoint-actor2.pth')
    torch.save(agent_duo.agent2.critic_local.state_dict(), 'weights/loosing-checkpoint-critic2.pth')

    return total_scores


def watch_agents(env: UnityEnvironment, brain_name: str, agent_duo: AgentDuo, episodes: int):
    """ Shows the agent simulation """

    for _ in range(episodes):
        env_info = env.reset(train_mode=False)[brain_name]
        # get the current state for each agent
        states = env_info.vector_observations
        # initialize the scores for the two agents
        scores = np.zeros(2)
        while True:
            action1 = agent_duo.agent1.act(states[0])
            action2 = agent_duo.agent2.act(states[1])
            # send both actions to the environment
            env_info = env.step([action1, action2])[brain_name]
            # get the next state (for each agent)
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            scores += rewards
            states = next_states
            if np.any(dones):
                break
        print(f"Scores: Agent 1: {scores[0]}, Agent 2: {scores[1]}")


def plot_scores(scores: dict, sma_window: int = 50) -> None:
    """
    Plots a line plot of the scores.
    The function expects the score of the first episode at scores[0] and the last episode at scores[-1]
    :param scores: a dictionary containing the scores for the first and second agent
    :param sma_window: Simple Moving Average rolling window
    :return:
    """
    # calculate moving average of the scores
    agent1_series: pd.Series = pd.Series(scores['agent1'])
    agent2_series: pd.Series = pd.Series(scores['agent2'])
    max_score_series: pd.Series = pd.Series(np.maximum(scores['agent1'], scores['agent2']))
    window1 = agent1_series.rolling(window=sma_window)
    window2 = agent2_series.rolling(window=sma_window)
    window3 = max_score_series.rolling(window=sma_window)
    agent1_scores_sma: pd.Series = window1.mean()
    agent2_scores_sma: pd.Series = window2.mean()
    max_score_sma: pd.Series = window3.mean()

    # plot the scores
    fig = plt.figure(figsize=(12, 5))

    plot1 = fig.add_subplot(121)
    plot1.plot(np.arange(len(scores['agent1'])), np.maximum(scores['agent1'], scores['agent2']))
    plot1.set_ylabel('Score')
    plot1.set_xlabel('Episode #')
    plot1.set_title("Raw scores")

    plot2 = fig.add_subplot(122)
    plot2.plot(np.arange(len(agent1_scores_sma)), agent1_scores_sma, c='blue', label='agent1')
    plot2.plot(np.arange(len(agent2_scores_sma)), agent2_scores_sma, c='red', label='agent2')
    plot2.plot(np.arange(len(max_score_sma)), max_score_sma, c='black', label='score')
    plot2.set_ylabel('Score')
    plot2.set_xlabel('Episode #')
    plot2.set_title(f"Moving Average(window={sma_window})")
    fig.legend()
    plt.show()


if __name__ == '__main__':

    # initialize the environment
    _env = UnityEnvironment(file_name="Tennis_Windows_x86_64/Tennis.exe")
    # _env = UnityEnvironment(file_name="Tennis_Linux/Tennis.x86_64")
    # get the default brain
    _brain_name = _env.brain_names[0]
    _brain = _env.brains[_brain_name]

    _agent_count: int = 2
    _action_size: int = 2
    _state_size: int = 24

    _agent1 = Agent(_state_size, _action_size,
                    gamma=0.994, lr_critic=0.0005, tau=0.001, weight_decay=0.)
    _agent2 = Agent(_state_size, _action_size,
                    gamma=0.994, lr_critic=0.0005, tau=0.001, weight_decay=0.)

    # set the same actor network for both agents
    _actor_local = ActorNetwork(_state_size, _action_size)
    _actor_target = ActorNetwork(_state_size, _action_size)
    _actor_optimizer = optim.Adam(_actor_local.parameters(), lr=0.001)
    _agent1.actor_target = _actor_target
    _agent2.actor_target = _actor_target
    _agent1.actor_local = _actor_local
    _agent2.actor_local = _actor_local
    _agent1.actor_optimizer = _actor_optimizer
    _agent2.actor_optimizer = _actor_optimizer
    _agent1.hard_update(_actor_local, _actor_target)
    _agent2.hard_update(_actor_local, _actor_target)

    # combine the two agents (this class will also store the shared ReplayBuffer)
    _agent_duo = AgentDuo(_agent1, _agent2, buffer_size=1000000, batch_size=150)

    watch_only = True
    if watch_only:
        watch_agents_from_pth_file(_env, _brain_name, _agent_duo, './weights')
    else:
        _scores = train_agents(_env, _brain_name, _agent_duo, n_episodes=2000)
        plot_scores(scores=_scores, sma_window=100)
        watch_agents(_env, _brain_name, _agent_duo, episodes=10)

    _env.close()
