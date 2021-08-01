import torch
from torch.nn import Linear, Module, MSELoss
from torch.optim import SGD, RMSprop, Adam
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# define linear function
# x = np.linspace(0, 20, 500)
# y = 5 * x +7
# plt.plot(x, y)
# plt.savefig('linear_function.pdf')
# plt.show()

x = np.random.rand(256)
noise = np.random.rand(256) / 4
y = x * 5 + 7 + noise

# df = pd.DataFrame()
# df['x'] = x
# df['y'] = y
# sns.lmplot(x='x', y='y', data=df)
# plt.savefig('linear_function_noise.pdf')
# plt.show()

model = Linear(1, 1)
criterion = MSELoss()
# optim = SGD(model.parameters(), lr=0.01)
# optim = RMSprop(model.parameters(), lr=0.01 ,alpha = 0.99)
optim = Adam(model.parameters(), lr=0.001, betas = (0.9, 0.999), eps = 1e-08)

epochs = 15001

x_train = x.reshape(-1, 1).astype('float32')
y_train = y.reshape(-1, 1).astype('float32')

for i in range(epochs):

	inputs = torch.from_numpy(x_train)
	lables = torch.from_numpy(y_train)

	outputs = model(inputs)

	optim.zero_grad()

	loss = criterion(outputs, lables)

	loss.backward()

	optim.step()
	if i % 100 == 0:
		print('epoch: {}, loss: {:1.4f}.'.format(i, loss.data.item()))

torch.save(model, 'linear_regression_pytorch.pkl')
# [w, b] = model.parameters()
# print(w.item(), b.item())

predicted = model.forward(torch.from_numpy(x_train)).data.numpy()
plt.scatter(x_train, y_train, c = 'blue', label = 'data', alpha =0.35)
plt.plot(x_train, predicted, c = 'red', label = 'predicted', alpha = 1)
plt.legend(loc = 'best', frameon = True)
plt.savefig('linear_regression_pytorch.pdf')
plt.show()
