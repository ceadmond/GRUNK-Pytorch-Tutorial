import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.fc1 = nn.Linear(1350, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        return x


net = Net()
# print(net)
"""
for name, parameters in net.named_parameters():
	print(parameters)
"""

input = torch.randn(1, 1, 32, 32)
out = net(input)
# print(input.size(), out.size())

# net.zero_grad()
# print(out.backward(torch.ones(1, 10)))

y = torch.arange(0, 10).view(1, 10).float()
criterion = nn.MSELoss()
loss = criterion(out, y)

# print(loss.item())

optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

optimizer.zero_grad()
loss.backward()

optimizer.step()