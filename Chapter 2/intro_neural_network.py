import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

flag = False
if flag:
	x = torch.linspace(-10, 10, 60)

	ax = plt.gca()
	ax.spines['right'].set_color('none')
	ax.spines['top'].set_color('none')

	ax.xaxis.set_ticks_position('bottom')
	ax.spines['bottom'].set_position(('data', 0))

	ax.yaxis.set_ticks_position('left')
	ax.spines['left'].set_position(('data', 0))

	plt.ylim((0, 1))
	sigmoid = torch.sigmoid(x)

	plt.plot(x.numpy(), sigmoid.numpy(), label='Sigmoid')
	plt.legend(loc='upper left', frameon=True)
	plt.savefig('sigmoid_function.pdf')
	plt.show()

flag = False
if flag:
	x = torch.linspace(-10, 10, 60)

	ax = plt.gca()
	ax.spines['right'].set_color('none')
	ax.spines['top'].set_color('none')

	ax.xaxis.set_ticks_position('bottom')
	ax.spines['bottom'].set_position(('data', 0))

	ax.yaxis.set_ticks_position('left')
	ax.spines['left'].set_position(('data', 0))

	plt.ylim((-1, 1))
	tanh = torch.tanh(x)

	plt.plot(x.numpy(), tanh.numpy(), label='tanh')
	plt.legend(loc='upper left', frameon=True)
	plt.savefig('tanh_function.pdf')
	plt.show()

flag = False
if flag:
	x = torch.linspace(-10, 10, 60)

	ax = plt.gca()
	ax.spines['right'].set_color('none')
	ax.spines['top'].set_color('none')

	ax.xaxis.set_ticks_position('bottom')
	ax.spines['bottom'].set_position(('data', 0))

	ax.yaxis.set_ticks_position('left')
	ax.spines['left'].set_position(('data', 0))

	plt.ylim((-3, 10))
	relu = F.relu(x)

	plt.plot(x.numpy(), relu.numpy(), label='ReLU')
	plt.legend(loc='upper left', frameon=True)
	plt.savefig('ReLU_function.pdf')
	plt.show()

flag = True
if flag:
	x = torch.linspace(-10, 10, 60)

	ax = plt.gca()
	ax.spines['right'].set_color('none')
	ax.spines['top'].set_color('none')

	ax.xaxis.set_ticks_position('bottom')
	ax.spines['bottom'].set_position(('data', 0))

	ax.yaxis.set_ticks_position('left')
	ax.spines['left'].set_position(('data', 0))

	plt.ylim((-3, 10))
	l_relu = F.leaky_relu(x, 0.1)

	plt.plot(x.numpy(), l_relu.numpy(), label='Leaky RuLU')
	plt.legend(loc='upper left', frameon=True)
	plt.savefig('Leaky_RuLU_function.pdf')
	plt.show()		