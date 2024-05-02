import torch
from agent.networks import CNN


class BCAgent:

    def __init__(self, learning_rate=1e-4, history_length=3):
        # define network, loss function, optimizer
        self.net = CNN(history_length)
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), learning_rate, weight_decay=1e-5
        )
        self.loss_function = torch.nn.CrossEntropyLoss()
        pass

    def update(self, X_batch, y_batch):
        # transform input to tensors
        X_batch = torch.from_numpy(X_batch).float()
        y_batch = torch.from_numpy(y_batch).long()

        # forward + backward + optimize
        outputs = self.net(X_batch)
        loss = self.loss_function(outputs, y_batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def predict(self, X):
        # forward pass
        X = torch.from_numpy(X).float()
        outputs = self.net(X)

        return outputs

    def load(self, file_name):
        self.net.load_state_dict(torch.load(file_name))

    def save(self, file_name):
        torch.save(self.net.state_dict(), file_name)
