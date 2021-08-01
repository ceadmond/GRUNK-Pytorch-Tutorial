import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (4, 3)
import numpy as np

transform = transforms.Compose(
	[transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5),  (0.5, 0.5, 0.5))])


trainset = torchvision.datasets.CIFAR10(		
	root='./data',
	train=True,
	download=True,
	transform=transform)

trainloader = torch.utils.data.DataLoader(
	trainset,
	batch_size=4,
	shuffle=True,
	num_workers=0)


testset = torchvision.datasets.CIFAR10(
	root='./data',
	train=False,
	download=True,
	transform=transform)

testloader = torch.utils.data.DataLoader(
	testset,
	batch_size=4,
	shuffle=False,
	num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
	'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class LeNet5(nn.Module):

	def __init__(self):
		super(LeNet5, self).__init__()

		self.conv1 = nn.Conv2d(3, 6, 5)
		self.conv2 = nn.Conv2d(6, 16, 5)

		self.fc1 = nn.Linear(16 * 5 * 5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
		x = F.max_pool2d(F.relu(self.conv2(x)), 2)
		x = x.view(-1, self.num_flat_features(x))
		x = F.leaky_relu(self.fc1(x), 0.01)
		x = F.leaky_relu(self.fc2(x), 0.01)
		x = self.fc3(x)
		return x

	def num_flat_features(self, x):
		size = x.size()[1:]
		num_features = 1
		for s in size:
			num_features *= s
		return num_features	 
		
net = LeNet5()
# print(net)	

# set loss function and optimizer		
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)
optimizer = optim.RMSprop(net.parameters(), lr=0.01 , alpha = 0.99)

# train the model
for epoch in range(2):

	running_loss = 0.0
	for i, data in enumerate(trainloader, 0):
		inputs, labels = data

		optimizer.zero_grad()

		outputs = net(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		running_loss += loss.item()
		if i % 2000 == 0:
			print('[%d, %5d] loss: %.3f' %
				(epoch + 1, i + 1, running_loss / 2000))
			running_loss = 0.0
		
print('Finished Training!')

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
	for data in testloader:
		images, labels = data
		outputs = net(images)
		_, predicted = torch.max(outputs, 1)
		c = (predicted == labels).squeeze()
		for i in range(4):
			label = labels[i]
			class_correct[label] += c[i].item()
			class_total[label] += 1
for i in range(10):
	print('Accuracy of %5s : %2d %%' % (
		classes[i], 100 * class_correct[i] / class_total[i]))

torch.save(net, 'cifar10_demo.pkl')
