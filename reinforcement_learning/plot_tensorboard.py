from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import matplotlib.pyplot as plt
import numpy as np
import os


def extract_data(event_file):
    event_acc = EventAccumulator(event_file)
    event_acc.Reload()  # load the data

    # Retrieve the scalars you're interested in
    data = {}
    for tag in event_acc.Tags()["scalars"]:
        x, y = [], []
        for scalar_event in event_acc.Scalars(tag):
            x.append(scalar_event.wall_time)
            y.append(scalar_event.value)
        data[tag] = (x, y)
    return data


# Specify the path to the TensorBoard logs
log_dir = "/home/lottv/Documents/DLL_Imitation_Reinforcement/reinforcement_learning/tensorboard/train/tensorboard_cartpole_results_2604_final"

# Get the path of the event file
event_file = next(iter(os.scandir(log_dir))).path

# Extract the data
data = extract_data(event_file)

# Plot the data
for tag, values in data.items():
    if tag == "mean_episode_reward":
        #     plt.figure(figsize=(10, 6))
        print(len(values[0]))
        episodes = range(len(values[0]))
        eval_cycle = 20
        new_range = [x * eval_cycle for x in episodes]
        plt.scatter(
            new_range,
            np.array(values[1]),
            s=10,
            c="b",
            marker="o",
            label="Episode Reward",
        )
        plt.plot(new_range, np.array(values[1]))
        plt.axhline(
            y=190,
            color="r",
        )

        plt.xlabel("Episode")
        plt.ylabel("Episode Reward")
        plt.title(f"TensorBoard Data: Mean Episode Reward")
        plt.legend()
        plt.grid(True)
        plt.show()

    # # Plot the data
    for tag, values in data.items():
        if tag == "episode_reward":
            # Generate 1000 equally spaced indices
            indices = np.linspace(0, len(values[1]) - 1, 1000, dtype=int)

            # Select the corresponding episode_reward values
            selected_values = np.array(values[1])[indices]

            plt.plot(range(len(selected_values)), selected_values)
            plt.axhline(
                y=190,
                color="r",
            )
            plt.xlabel("Episode")
            plt.ylabel("Episode Reward")
            plt.title(f"TensorBoard Data: Episode Reward")
            plt.grid(True)
            plt.show()

    if tag == "a_1":
        # Generate 1000 equally spaced indices
        indices = np.linspace(0, len(values[1]) - 1, 1000, dtype=int)

        # Select the corresponding a_0 values
        selected_values = np.array(values[1])[indices]

        # plt.figure(figsize=(10, 6))
        plt.plot(range(len(selected_values)), selected_values)
        plt.xlabel("Episode")
        plt.ylabel("Action Usage (a_1)")
        plt.title("TensorBoard Data: Action Usage (a_1)")
        plt.grid(True)

        plt.show()

# if __name__=="__main__":
#
#     # Load the TensorBoard log
#     event_acc = EventAccumulator('/home/lottv/Documents/DLL_Imitation_Reinforcement/imitation_learning/tensorboard/agent_hist3_2704_2')
#     event_acc.Reload()
#
#     # Extract the metrics
#     train_loss = event_acc.Scalars('loss')
#     valid_loss = event_acc.Scalars('valid_loss')
#     train_acc = event_acc.Scalars('train_accuracy')
#     valid_acc = event_acc.Scalars('valid_accuracy')
#     train_f1 = event_acc.Scalars('train_f1')
#     valid_f1 = event_acc.Scalars('valid_f1')
#
#     # Plot the loss
#     plt.plot([e.step for e in train_loss], [e.value for e in train_loss], label='Train Loss')
#     plt.plot([e.step for e in valid_loss], [e.value for e in valid_loss], label='Valid Loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid(True)  # Add grid
#
#     plt.show()
#
#     # Plot the accuracy
#     plt.plot([e.step for e in train_acc], [e.value for e in train_acc], label='Train Accuracy')
#     plt.plot([e.step for e in valid_acc], [e.value for e in valid_acc], label='Valid Accuracy')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.legend()
#     plt.grid(True)  # Add grid
#
#     plt.show()
#
#     # Plot the F1 score
#     plt.plot([e.step for e in train_f1], [e.value for e in train_f1], label='Train F1')
#     plt.plot([e.step for e in valid_f1], [e.value for e in valid_f1], label='Valid F1')
#     plt.xlabel('Epochs')
#     plt.ylabel('F1 Score')
#     plt.legend()
#     plt.grid(True)  # Add grid
#
#     plt.show()
