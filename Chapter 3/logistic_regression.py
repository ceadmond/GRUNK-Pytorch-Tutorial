import torch
import torch.nn as nn
import matplotlib.pyplot as plt
# from matplotlib import animation
import numpy as np
torch.manual_seed(10)

# creat data
sample_nums = 10
mean_value = 1.7
bias = 1
n_data = torch.ones(sample_nums, 2)
x0 = torch.normal(mean_value * n_data, 1) + bias
y0 = torch.zeros(sample_nums)
x1 = torch.normal(- mean_value * n_data, 1) + bias
y1 = torch.ones(sample_nums)
train_x = torch.cat((x0, x1), 0)
train_y = torch.cat((y0, y1), 0)

# select model
class LR(nn.Module):
	def __init__(self):
		super(LR, self).__init__()
		self.features = nn.Linear(2, 1)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		x = self.features(x)
		x = self.sigmoid(x)
		return x

lr_net = LR()

loss_fn = nn.BCELoss()

optimizer = torch.optim.SGD(lr_net.parameters(), lr=0.01, momentum=0.9)

for iteration in range(1001):
	y_pred = lr_net(train_x)

	loss = loss_fn(y_pred.squeeze(), train_y)

	loss.backward()
    # update parameters
	optimizer.step()

	# plot
	if iteration % 20 ==0:
		mask = y_pred.ge(0.5).float().squeeze()
		correct = (mask == train_y).sum()
		acc = correct.item() / train_y.size(0)

		plt.scatter(x0.data.numpy()[:, 0], x0.data.numpy()[:, 1], c='r', label='class 0')
		plt.scatter(x1.data.numpy()[:, 0], x1.data.numpy()[:, 1], c='b', label='class 1')

		w0, w1 = lr_net.features.weight[0]
		w0, w1 = float(w0.item()), float(w1.item())
		plot_b = float(lr_net.features.bias[0].item())
		plot_x = np.arange(-6, 6, 0.1)
		plot_y = (-w0 * plot_x - plot_b) / w1

		plt.xlim(-5, 7)
		plt.ylim(-7, 7)
		plt.plot(plot_x, plot_y)

		plt.text(-5, 5, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size':20, 'color':'red'})
		plt.title('Iteration: {} w0: {:0.2f} w1: {:.2f} b:{:.2f} accuracy: {:.2%}'.format(iteration, w0, w1, plot_b, acc))
		plt.legend(loc='best', frameon=True)

		plt.show()
		plt.pause(0.5)
		# animation.save('logistic_regression.gif', writer='imagemagick')
		# plt.savefig('logistic_regression.pdf')

		if acc > 0.99:
			break
