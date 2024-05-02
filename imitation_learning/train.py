import sys

sys.path.append("../")
import pickle
import numpy as np
import os
import gzip
import matplotlib.pyplot as plt
import torch
from torch.utils.data import WeightedRandomSampler, TensorDataset, DataLoader
from sklearn.metrics import f1_score

import utils
from agent.bc_agent import BCAgent
from tensorboard_evaluation import Evaluation


def balance_data(X_train, y_train):
    def plot_data_dist(
        y_all,
    ):
        unique_all, counts_all = np.unique(y_all, return_counts=True)
        plt.figure(figsize=(6, 6))
        plt.bar(unique_all, counts_all)
        # plt.title('Class distribution in Dataset')
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.xticks(unique_all)
        plt.show()

    # check and plot imbalance
    plot_data_dist(y_train)

    # compute class weights
    unique_classes, counts_all = np.unique(y_train, return_counts=True)
    class_weights = 1 / counts_all
    sample_weights = class_weights[np.searchsorted(unique_classes, y_train)]

    # create and apply WeightedRandomSampler
    X_train_tensor = torch.from_numpy(X_train)
    y_train_tensor = torch.from_numpy(y_train)
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(y_train), replacement=True
    )
    dl = DataLoader(dataset, sampler=sampler)

    # extract balanced data
    X_train_balanced = []
    y_train_balanced = []
    for X, y in dl:
        X_train_balanced.append(X.numpy())
        y_train_balanced.append(y.numpy())
    X_train = np.concatenate(X_train_balanced, axis=0)
    y_train = np.concatenate(y_train_balanced, axis=0)

    plot_data_dist(y_train)

    return X_train, y_train


def read_data(datasets_dir="./data", frac=0.1):
    """
    This method reads the states and actions recorded in drive_manually.py
    and splits it into training/ validation set.
    """
    print("... read data")
    data_file = os.path.join(datasets_dir, "data_v1_2504.pkl.gzip")

    f = gzip.open(data_file, "rb")
    data = pickle.load(f)

    # get images as features and actions as targets
    X = np.array(data["state"][:20000]).astype("float32")
    y = np.array(data["action"][:20000]).astype("float32")

    # split data into training and validation set
    n_samples = len(data["state"][:20000])
    print(f"n_samples: {n_samples}")
    X_train, y_train = (
        X[: int((1 - frac) * n_samples)],
        y[: int((1 - frac) * n_samples)],
    )
    X_valid, y_valid = (
        X[int((1 - frac) * n_samples) :],
        y[int((1 - frac) * n_samples) :],
    )
    return X_train, y_train, X_valid, y_valid


def preprocessing(X_train, y_train, X_valid, y_valid, hist_len, balancing=False):

    # X = images
    X_train = utils.rgb2gray(X_train)
    X_valid = utils.rgb2gray(X_valid)
    # y = actions
    y_train = np.array([utils.action_to_id(i) for i in y_train])
    y_valid = np.array([utils.action_to_id(i) for i in y_valid])

    # History:
    def add_history(X, window_size):
        num_samples, height, width = X.shape
        X_with_history = np.zeros((num_samples, height, width, window_size))
        for i in range(num_samples):
            for j in range(min(i + 1, window_size)):
                X_with_history[i, :, :, j] = X[i - j, :, :]
        return np.transpose(X_with_history, (0, 3, 1, 2))

    X_train = add_history(X_train, hist_len)
    X_valid = add_history(X_valid, hist_len)

    # Check if sliding window is working correctly
    if hist_len > 1:
        for i in range(len(X_train) - 1):
            if not np.array_equal(X_train[i + 1][1, :, :], X_train[i][0, :, :]):
                print(f"Sliding window is incorrect for sample {i}.")
                break

    if balancing:
        X_train, y_train = balance_data(X_train, y_train)

    print(f"training data: {len(y_train)}")
    print(f"validation data: {len(y_valid)}")
    print(f"Ratio of training data: {len(y_train) / len(y_valid)}")

    return X_train, y_train, X_valid, y_valid


def train_model(
    X_train,
    y_train,
    X_valid,
    y_valid,
    hist_len,
    n_minibatches,
    batch_size,
    lr,
    model_dir="./models",
    tensorboard_dir="./tensorboard",
):
    # create result and model folders
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    print("... train model")

    # specify agent
    agent = BCAgent(lr, hist_len)

    tensorboard_eval = Evaluation(
        tensorboard_dir,
        "Imitation Learning",
        stats=[
            "loss",
            "valid_loss",
            "train_accuracy",
            "valid_accuracy",
            "train_f1",
            "valid_f1",
        ],
    )

    def sample_minibatch(X, y, b_size):
        indices = np.random.choice(len(X), b_size, replace=False)
        X_b = X[indices]
        y_b = y[indices]
        return X_b, y_b

    for i in range(n_minibatches):
        X_batch, y_batch = sample_minibatch(X_train, y_train, batch_size)
        loss = agent.update(X_batch, y_batch)

        if i % 10 == 0:
            # compute training/ validation metrics and write them to tensorboard
            y_train_pred = agent.predict(X_train)
            y_valid_pred = agent.predict(X_valid)
            valid_loss = agent.loss_function(y_valid_pred, torch.from_numpy(y_valid))
            train_acc = np.mean(
                np.argmax(y_train_pred.detach().numpy(), axis=1) == y_train
            )
            valid_acc = np.mean(
                np.argmax(y_valid_pred.detach().numpy(), axis=1) == y_valid
            )
            train_f1 = f1_score(
                y_train,
                np.argmax(y_train_pred.detach().numpy(), axis=1),
                average="weighted",
            )
            valid_f1 = f1_score(
                y_valid,
                np.argmax(y_valid_pred.detach().numpy(), axis=1),
                average="weighted",
            )
            tensorboard_eval.write_episode_data(
                i,
                {
                    "loss": loss,
                    "valid_loss": valid_loss.item(),
                    "train_accuracy": train_acc,
                    "valid_accuracy": valid_acc,
                    "train_f1": train_f1,
                    "valid_f1": valid_f1,
                },
            )

    # TODO: save your agent
    model_dir = agent.save(os.path.join(model_dir, "agent.pt"))
    print("Model saved in file: %s" % model_dir)


if __name__ == "__main__":
    # set history length
    history_length = 1
    print("History length: ", history_length)

    # read data
    X_train, y_train, X_valid, y_valid = read_data("./data")

    # preprocess data
    X_train, y_train, X_valid, y_valid = preprocessing(
        X_train, y_train, X_valid, y_valid, hist_len=history_length, balancing=True
    )

    # train model (you can change the parameters!)
    train_model(
        X_train,
        y_train,
        X_valid,
        y_valid,
        hist_len=history_length,
        n_minibatches=1000,
        batch_size=64,
        lr=1e-4,
    )
