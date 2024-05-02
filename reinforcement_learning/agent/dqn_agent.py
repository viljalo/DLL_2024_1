import sys

sys.path.append("../")
import numpy as np
import torch
import torch.optim as optim
from agent.replay_buffer import ReplayBuffer


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class DQNAgent:

    def __init__(
        self,
        Q,
        Q_target,
        num_actions,
        gamma=0.95,
        batch_size=64,
        epsilon=0.1,
        tau=0.01,
        lr=1e-3,
        history_length=500,
    ):
        """
        Q-Learning agent for off-policy TD control using Function Approximation.
        Finds the optimal greedy policy while following an epsilon-greedy policy.

        Args:
           Q: Action-Value function estimator (Neural Network)
           Q_target: Slowly updated target network to calculate the targets.
           num_actions: Number of actions of the environment.
           gamma: discount factor of future rewards.
           batch_size: Number of samples per batch.
           tau: indicates the speed of adjustment of the slowly updated target network.
           epsilon: Chance to sample a random action. Float betwen 0 and 1.
           lr: learning rate of the optimizer
        """
        # setup networks
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.Q = Q.to(device)
        # self.Q_target = Q_target.to(device)
        # self.Q_target.load_state_dict(self.Q.state_dict())
        # #
        self.Q = Q.cuda()
        self.Q_target = Q_target.cuda()
        self.Q_target.load_state_dict(self.Q.state_dict())

        # define replay buffer
        self.replay_buffer = ReplayBuffer(history_length)

        # parameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon

        self.loss_function = torch.nn.MSELoss()
        self.optimizer = optim.Adam(self.Q.parameters(), lr=lr)

        self.num_actions = num_actions

    def train(self, state, action, next_state, reward, terminal):
        """
        This method stores a transition to the replay buffer and updates the Q networks.
        """

        # TODO:
        # 1. add current transition to replay buffer
        # 2. sample next batch and perform batch update:
        #       2.1 compute td targets and loss
        #              td_target =  reward + discount * max_a Q_target(next_state_batch, a)
        #       2.2 update the Q network
        #       2.3 call soft update for target network
        #           soft_update(self.Q_target, self.Q, self.tau)

        # add current transition to replay buffer
        self.replay_buffer.add_transition(
            state, action, next_state, reward, terminal
        )  # done = terminal ??

        # sample next batch and perform batch update:
        states, actions, next_states, rewards, terminals = (
            self.replay_buffer.next_batch(self.batch_size)
        )

        # convert to tensors
        states = torch.from_numpy(states).to(self.device).float()
        actions = torch.from_numpy(actions).to(self.device).float()
        next_states = torch.from_numpy(next_states).to(self.device).float()
        rewards = torch.from_numpy(rewards).to(self.device).float()
        terminals = torch.from_numpy(terminals).to(self.device).float()

        # compute td targets  (r_t+1 + gamma * max_a Q_target(s_t+1, a))
        with torch.no_grad():  # save computational resources
            next_state_batch = self.Q_target(next_states)  # Q-values of next states
            max_a_Q_targets = torch.max(next_state_batch, dim=1)[
                0
            ]  # max Q-value of next state
            td_targets = rewards + max_a_Q_targets * (
                1 - terminals.float()
            )  # handle terminal states

        # compute current q values for given states and actions
        actions = actions.to(self.device).long()
        curr_q_values = (
            self.Q.forward(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        )  # match the shape of td_targets

        # compute loss and update the Q network
        loss = self.loss_function(
            curr_q_values, td_targets
        )  # loss between current Q-values and TD targets (MSE)
        self.optimizer.zero_grad()  # clear gradients from previous training step
        loss.backward()  # backpropagate the loss
        self.optimizer.step()  # update the weights of Q network

        # call soft update for target network
        soft_update(self.Q_target, self.Q, self.tau)

        return loss

    def act(self, state, deterministic, env="CartPole-v0"):
        """
        This method creates an epsilon-greedy policy based on the Q-function approximator and epsilon (probability to select a random action)
        Args:
            state: current state input
            deterministic:  if True, the agent should execute the argmax action (False in training, True in evaluation)
        Returns:
            action id
        """
        r = np.random.uniform()
        if deterministic or r > self.epsilon:
            # TODO: take greedy action (argmax)
            if env == "CarRacing-v0":
                action_id = torch.argmax(
                    self.Q(
                        torch.Tensor(state)
                        .type(torch.FloatTensor)
                        .unsqueeze(0)
                        .to(self.device)
                    )
                ).item()

            else:
                action_id = torch.argmax(
                    self.Q(
                        torch.Tensor(state)
                        .type(torch.FloatTensor)
                        .unsqueeze(0)
                        .to(self.Q.device)
                    )
                ).item()
        else:
            # TODO: sample random action
            # Hint for the exploration in CarRacing: sampling the action from a uniform distribution will probably not work.
            # You can sample the agents actions with different probabilities (need to sum up to 1) so that the agent will prefer to accelerate or going straight.
            # To see how the agent explores, turn the rendering in the training on and look what the agent is doing.
            if env == "CarRacing-v0":
                action_probabilities = [0.4, 0.13, 0.13, 0.3, 0.04]
                assert np.sum(action_probabilities) == 1
                action_id = np.random.choice(self.num_actions, p=action_probabilities)
            else:
                action_id = np.random.choice(self.num_actions)  # sample random action

        return action_id

    def save(self, file_name):
        torch.save(self.Q.state_dict(), file_name)

    def load(self, file_name):
        self.Q.load_state_dict(torch.load(file_name))
        self.Q_target.load_state_dict(torch.load(file_name))
