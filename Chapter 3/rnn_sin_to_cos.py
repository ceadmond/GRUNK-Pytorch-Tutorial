import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math
import random


TIME_STEP = 10
INPUT_SIZE = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
H_SIZE = 64
EPOCHS = 300
h_state = None

steps = np.linspace(0, np.pi * 2, 256, dtype=np.float32)
x_np = np.sin(steps)
y_np = np.cos(steps)

"""
plt.figure(1)
plt.suptitle("Sin and Cos", fontsize="18")
plt.plot(steps, x_np, "r-", label="target (cos)")
plt.plot(steps, y_np, "b-", label="input (sin)")
plt.legend(loc="best", frameon=True)
plt.savefig("sin_cos_function.pdf")
plt.show()
"""


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=H_SIZE,
            num_layers=1,
            batch_first=True,
        )
        self.out = nn.Linear(H_SIZE, 1)

    def forward(self, x, h_state):
        r_out, h_state = self.rnn(x, h_state)
        outs = []
        for time_step in range(r_out.size(1)):
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1), h_state


rnn = RNN().to(DEVICE)
optimizer = torch.optim.Adam(rnn.parameters())
criterion = nn.MSELoss()

rnn.train()
fig2 = plt.figure(2)
for step in range(EPOCHS):
    start, end = step * np.pi, (step + 1) * np.pi
    steps = np.linspace(start, end, TIME_STEP, dtype=np.float32)
    x_np = np.sin(steps)
    y_np = np.cos(steps)
    x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])
    y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])
    x = x.to(DEVICE)
    prediction, h_state = rnn(x, h_state)
    h_state = h_state.data
    loss = criterion(prediction.cpu(), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (step + 1) % 20 == 0:
        print("EPOCHS: {}, Loss: {:.4f}".format(step, loss))
        plt.plot(steps, y_np.flatten(), "r-")
        plt.plot(steps, prediction.cpu().data.numpy().flatten(), "b-")
        plt.draw()
        plt.pause(0.001)
