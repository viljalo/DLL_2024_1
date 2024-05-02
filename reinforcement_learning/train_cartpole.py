import sys

sys.path.append("../")

import numpy as np
import gym
import itertools as it
from agent.dqn_agent import DQNAgent
from tensorboard_evaluation import *
from agent.networks import MLP
from utils import EpisodeStats
import torch


def run_episode(
    env, agent, deterministic, do_training=True, rendering=False, max_timesteps=200
):
    """
    This methods runs one episode for a gym environment.
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """

    stats = EpisodeStats()  # save statistics like episode reward or action usage
    state = env.reset()

    step = 0
    loss = None
    while True:

        action_id = agent.act(state=state, deterministic=deterministic)
        next_state, reward, terminal, info = env.step(action_id)

        if do_training:
            loss = agent.train(state, action_id, next_state, reward, terminal)

        stats.step(reward, action_id, loss)

        state = next_state

        if rendering:
            env.render()

        if terminal or step > max_timesteps:
            break

        step += 1

    return stats


def train_online(
    env,
    agent,
    num_episodes,
    model_dir="./models_cartpole",
    tensorboard_dir="./tensorboard",
):
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    print("... train agent")

    tensorboard = Evaluation(
        os.path.join(tensorboard_dir, "train"),
        name="test",
        stats=["episode_reward", "a_0", "a_1", "mean_episode_reward", "loss"],
    )
    problem_solved = 0
    # training
    for i in range(num_episodes):
        stats = run_episode(env, agent, deterministic=False, do_training=True)
        tensorboard.write_episode_data(
            i,
            eval_dict={
                "episode_reward": stats.episode_reward,
                "a_0": stats.get_action_usage(0),
                "a_1": stats.get_action_usage(1),
                "loss": np.mean(stats.episode_losses),
            },
        )

        # TODO: evaluate your agent every 'eval_cycle' episodes using run_episode(env, agent, deterministic=True, do_training=False) to
        if i % eval_cycle == 0 or i >= (num_episodes - 1):
            total_reward = 0
            for j in range(num_eval_episodes):
                stats = run_episode(env, agent, deterministic=True, do_training=False)
                total_reward += stats.episode_reward
                tensorboard.write_episode_data(
                    i,
                    eval_dict={
                        "episode_reward": stats.episode_reward,
                        "a_0": stats.get_action_usage(0),
                        "a_1": stats.get_action_usage(1),
                    },
                )
            mean_reward = total_reward / num_eval_episodes
            print(f"Mean reward: {mean_reward}")
            tensorboard.write_episode_data(
                i, eval_dict={"mean_episode_reward": mean_reward}
            )

        # Store model.
        if i % eval_cycle == 0 or i >= (num_episodes - 1):
            agent.save(os.path.join(model_dir, "dqn_agent.pt"))

    tensorboard.close_session()


if __name__ == "__main__":
    print(torch.cuda.is_available())
    num_eval_episodes = 5  # evaluate on 5 episodes
    eval_cycle = 20  # evaluate every 20 episodes

    env = gym.make("CartPole-v0").unwrapped

    state_dim = 4
    num_actions = 2

    # Q-function is represented by a multi-layer perceptron
    # state_dim consists of cart pos., cart velocity, pole angle, pole velocity
    # action_dim consists of left or right
    Q = MLP(state_dim=4, action_dim=2)

    # Target network is identical to the Q network
    # Q_target is used to compute the target Q-values
    Q_target = MLP(state_dim=4, action_dim=2)

    # Initialize the DQN agent
    agent = DQNAgent(
        Q,
        Q_target,
        num_actions,
        gamma=0.96,
        batch_size=64,
        epsilon=0.1,
        tau=0.01,
        lr=1e-3,
        history_length=500,
    )

    train_online(env, agent, 1000)
