import sys

sys.path.append("../")
import gym
from tensorboard_evaluation import *
from utils import *
from agent.dqn_agent import DQNAgent
from imitation_learning.agent.networks import CNN


def run_episode(
    env,
    agent,
    deterministic,
    skip_frames=2,
    do_training=True,
    rendering=False,
    max_timesteps=1000,
    history_length=30,
):
    """
    This methods runs one episode for a gym environment.
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """
    # print("run episode")

    stats = EpisodeStats()

    # Save history
    image_hist = []

    step = 0
    state = env.reset()

    # fix bug of corrupted states without rendering in gym environment
    env.viewer.window.dispatch_events()

    # append image history to first state
    state = state_preprocessing(state)
    image_hist.extend([state] * history_length)
    state = np.array(image_hist).reshape(history_length, 96, 96)

    while True:

        # TODO: get action_id from agent
        # Hint: adapt the probabilities of the 5 actions for random sampling so that the agent explores properly.
        action_id = agent.act(
            state=state, deterministic=deterministic, env="CarRacing-v0"
        )
        action = id_to_action(action_id)

        reward = 0

        # Hint: frame skipping might help you to get better results.
        for _ in range(skip_frames + 1):
            next_state, r, terminal, info = env.step(action)
            reward += r

            if rendering:
                env.render()

            if terminal:
                break

        next_state = state_preprocessing(next_state)
        image_hist.append(next_state)
        image_hist.pop(0)
        next_state = np.array(image_hist).reshape(history_length, 96, 96)

        if do_training:
            agent.train(state, action_id, next_state, reward, terminal)

        stats.step(reward, action_id, None)

        state = next_state

        if terminal or (step * (skip_frames + 1)) > max_timesteps:
            break

        step += 1

    return stats


def train_online(
    env,
    agent,
    num_episodes,
    history_length=25,
    model_dir="./models_carracing",
    tensorboard_dir="./tensorboard",
):

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    print("... train agent")
    tensorboard = Evaluation(
        os.path.join(tensorboard_dir, "train_carracing"),
        name="train_carracing",
        stats=[
            "episode_reward",
            "straight",
            "left",
            "right",
            "accel",
            "brake",
            "average_reward",
        ],
    )

    best_eval = 0
    decay_episodes = 200
    epsilon_start = 0.9
    epsilon_end = 0.1
    max_timesteps_start = 100
    max_timesteps_end = 750

    for i in range(num_episodes):
        print("epsiode %d" % i)

        if i < decay_episodes:
            epsilon_decay = (epsilon_start - epsilon_end) / decay_episodes
            max_timesteps_decay = (
                max_timesteps_end - max_timesteps_start
            ) / decay_episodes
            agent.epsilon = epsilon_start - i * epsilon_decay
            max_timesteps = int(max_timesteps_start + i * max_timesteps_decay)
        else:
            agent.epsilon = epsilon_end
            max_timesteps = max_timesteps_end

        print(f"Epsilon: {agent.epsilon}")
        print(f"Max timesteps: {max_timesteps}")

        # Hint: you can keep the episodes short in the beginning by changing max_timesteps (otherwise the car will spend most of the time out of the track)

        stats = run_episode(
            env,
            agent,
            max_timesteps=max_timesteps,
            deterministic=False,
            do_training=True,
            rendering=True,
        )

        tensorboard.write_episode_data(
            i,
            eval_dict={
                "episode_reward": stats.episode_reward,
                "straight": stats.get_action_usage(STRAIGHT),
                "left": stats.get_action_usage(LEFT),
                "right": stats.get_action_usage(RIGHT),
                "accel": stats.get_action_usage(ACCELERATE),
                "brake": stats.get_action_usage(BRAKE),
            },
        )
        print("Episode reward: ", stats.episode_reward)

        # TODO: evaluate your agent every 'eval_cycle' episodes using run_episode(env, agent, deterministic=True, do_training=False) to
        # check its performance with greedy actions only. You can also use tensorboard to plot the mean episode reward.

        # store model.
        if i % eval_cycle == 0 or (i >= num_episodes - 1):
            print("Evaluating agent ...")
            average_reward = 0
            total_reward = 0
            for episode in range(num_eval_episodes):
                episode_stats = run_episode(
                    env,
                    agent,
                    deterministic=True,
                    do_training=False,
                    rendering=True,
                    skip_frames=2,
                    history_length=history_length,
                    max_timesteps=max_timesteps,
                )
                print(f"History length: {history_length}")
                average_reward = (
                    average_reward * episode + episode_stats.episode_reward
                ) / (episode + 1)
                total_reward += episode_stats.episode_reward
            if average_reward > best_eval:  # save best model
                best_eval = average_reward
                model_path = os.path.join(model_dir, "dqn_agent_best.pt")
                agent.save(model_path)
                print(f"New best model saved at {model_path}")
            avg_reward = total_reward / num_eval_episodes
            tensorboard.write_episode_data(i, {"average_reward": avg_reward})

        agent.save(os.path.join(model_dir, "dqn_agent.pt"))

    tensorboard.close_session()


def state_preprocessing(state):
    return rgb2gray(state).reshape(96, 96) / 255.0


if __name__ == "__main__":

    num_eval_episodes = 5
    eval_cycle = 20
    history_length = 30
    num_of_actions = 5

    env = gym.make("CarRacing-v0").unwrapped

    # define Q network, target network and DQN agent
    Q = CNN(history_length=history_length, n_classes=num_of_actions)
    Q_target = CNN(history_length=history_length, n_classes=num_of_actions)
    agent = DQNAgent(Q, Q_target, num_of_actions, history_length=history_length)

    train_online(
        env,
        agent,
        num_episodes=1000,
        history_length=history_length,
        model_dir="./models_carracing",
    )
