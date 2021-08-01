import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# load data
data = np.loadtxt("german.data-numeric")

# data normalization
n, l = data.shape
for j in range(l - 1):
    meanVal = np.mean(data[:, j])
    stdVal = np.std(data[:, j])
    data[:, j] = (data[:, j] - meanVal) / stdVal

# shuffle data
np.random.shuffle(data)

# train_data and test_data
train_data = data[:900, : l - 1]
train_lab = data[:900, l - 1] - 1

test_data = data[900:, : l - 1]
test_lab = data[900:, l - 1] - 1

# define the model
class LR(nn.Module):
    def __init__(self):
        super(LR, self).__init__()
        self.fc1 = nn.Linear(24, 2)

    def forward(self, x):
        out = F.tanh(self.fc1(x))

        return out


def test(pred, lab):
    t = pred.max(-1)[1] == lab
    return torch.mean(t.float())


net = LR()

criterion = nn.CrossEntropyLoss()
optim = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
# optim = optim.RMSprop(net.parameters(), lr=0.05, alpha = 0.75)
# optim = optim.Adam(net.parameters(), lr=0.001, betas = (0.9, 0.999), eps = 1e-08)
epochs = 8001

for i in range(epochs):
    net.train()

    x = torch.from_numpy(train_data).float()
    y = torch.from_numpy(train_lab).long()
    y_hat = net(x)

    loss = criterion(y_hat, y)
    optim.zero_grad()
    loss.backward()
    optim.step()

    if (i + 1) % 100 == 0:
        net.eval()
        test_in = torch.from_numpy(test_data).float()
        test_l = torch.from_numpy(test_lab).long()
        test_out = net(test_in)

        acc = test(test_out, test_l)
        print(
            "Epoch: {}, Loss: {:.2%}, Accuracy: {:.2%}".format(i + 1, loss.item(), acc)
        )

torch.save(net, "logistic_uci_german.pkl")
