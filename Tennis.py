from unityagents import UnityEnvironment
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from Agent import Agent, AgentDuo


def watch_agents_from_pth_file(_env, _brain_name, _agent_duo, param, param1):
    pass


def train_agents(_env, _brain_name, _agent_duo, n_episodes, max_steps):
    pass


def watch_agents(env: UnityEnvironment, brain_name: str, agent_duo: AgentDuo):
    """ Shows the agent simulation """
    env_info = env.reset(train_mode=False)[brain_name]
    # get the current state for each agent
    states = env_info.vector_observations
    # initialize the scores for the two agents
    scores = np.zeros(2)

    while True:
        action1 = agent_duo.agent1.act(states)
        action2 = agent_duo.agent2.act(states)
        # send both actions to the environment
        env_info = env.step(action1, action2)[brain_name]
        # get the next state (for each agent)
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        scores += rewards
        states = next_states
        if np.any(dones):
            break
    print(f"Scores: Agent 1: {scores[0]}, Agent 2: {scores[1]}")


def plot_scores(scores: [int], sma_window: int = 50) -> None:
    """
    Plots a line plot of the scores.
    The function expects the score of the first episode at scores[0] and the last episode at scores[-1]
    :param scores:
    :param sma_window: Simple Moving Average rolling window
    :return:
    """
    # calculate moving average of the scores
    series: pd.Series = pd.Series(scores)
    window = series.rolling(window=sma_window)
    scores_sma: pd.Series = window.mean()

    # plot the scores
    fig = plt.figure(figsize=(12, 5))

    plot1 = fig.add_subplot(121)
    plot1.plot(np.arange(len(scores)), scores)
    plot1.set_ylabel('Score')
    plot1.set_xlabel('Episode #')
    plot1.set_title("Raw scores")

    plot2 = fig.add_subplot(122)
    plot2.plot(np.arange(len(scores_sma)), scores_sma)
    plot2.set_ylabel('Score')
    plot2.set_xlabel('Episode #')
    plot2.set_title(f"Moving Average(window={sma_window})")

    plt.show()


if __name__ == '__main__':
    # initialize the environment
    _env = UnityEnvironment(file_name="Tennis_Windows_x86_64/Tennis.exe")
    # get the default brain
    _brain_name = _env.brain_names[0]
    _brain = _env.brains[_brain_name]

    _agent_count: int = 2
    _action_size: int = 2
    _state_size: int = 24

    _agent1 = Agent(_state_size, _action_size, gamma=0.99, lr_actor=0.0002, lr_critic=0.0003, tau=0.002, weight_decay=0.0001)
    _agent2 = Agent(_state_size, _action_size, gamma=0.99, lr_actor=0.0002, lr_critic=0.0003, tau=0.002, weight_decay=0.0001)

    _agent_duo = AgentDuo(_agent1, _agent2, buffer_size=1000000, batch_size=128)

    watch_only = False
    if watch_only:
        watch_agents_from_pth_file(_env, _brain_name, _agent_duo, './checkpoint-actor.pth', './checkpoint-critic.pth')
    else:
        # scores = train_agents(_env, _brain_name, _agent_duo, n_episodes=500, max_steps=1500)
        watch_agents(_env, _brain_name, _agent_duo)
        # plot_scores(scores=scores, sma_window=10)

    _env.close()
