import sys

sys.path.append("../")
sys.path.append(".")
from datetime import datetime
import numpy as np
import gym
import os
import json
import torch
import torch.nn.functional as F

from agent.bc_agent import BCAgent
from utils import *


def run_episode(env, agent, history_length, rendering=True, max_timesteps=1000):

    episode_reward = 0
    step = 0
    history_length = history_length
    state_hist = None
    state = env.reset()

    # fix bug of curropted states without rendering in racingcar gym environment
    env.viewer.window.dispatch_events()

    while True:
        # preprocess the state in the same way than in your preprocessing in train_agent.py
        state = rgb2gray(state)

        if step == 0:
            state_with_history = np.zeros((1, 96, 96, history_length))
            state_with_history[:, :, :, 0] = state
        else:
            state_with_history = state_hist
            state_with_history[:, :, :, 1:] = state_with_history[:, :, :, :-1]
            state_with_history[:, :, :, 0] = state

        state_hist = state_with_history
        state = np.transpose(state_with_history, (0, 3, 1, 2))

        # get the most probable action and transform it from discretized action to continuous action.
        act = agent.predict(state)
        act = torch.argmax(F.softmax(act), dim=1)
        a = id_to_action(act, max_speed=2)

        # give some acceleration at the beginning
        if step <= 10:
            a = np.array([0.0, 1.0, 0.0])

        next_state, r, done, info = env.step(a)
        episode_reward += r
        state = next_state
        step += 1

        if rendering:
            env.render()

        if done or step > max_timesteps:
            break

    return episode_reward


if __name__ == "__main__":

    # important: don't set rendering to False for evaluation (you may get corrupted state images from gym)
    rendering = True

    n_test_episodes = 15  # number of episodes to test

    history_length = 3

    # load agent
    agent = BCAgent(history_length=history_length)
    agent.load("models/agent_hist3_2704_2.pt")

    env = gym.make("CarRacing-v0").unwrapped


    episode_rewards = []
    for i in range(n_test_episodes):
        episode_reward = run_episode(
            env, agent, history_length=history_length, rendering=rendering
        )
        episode_rewards.append(episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()

    fname = "results/results_bc_agent-%s.json" % datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )
    fh = open(fname, "w")
    json.dump(results, fh)

    env.close()
    print("... finished")
