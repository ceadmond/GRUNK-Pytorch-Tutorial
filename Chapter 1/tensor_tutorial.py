import numpy as np
import torch

"""
x = torch.empty(5, 3)
print(x)
"""

"""
x = torch.rand(5, 3)
print(x)
"""

"""
x = torch.zeros(5, 3, dtype=torch.long)
print(x)
"""

"""
x = torch.tensor([5.5, 3])
print(x)
"""

"""
x =  torch.zeros(5, 3, dtype=torch.long)
x = x.new_ones(5, 3, dtype=torch.double)
print(x)

x = torch.randn_like(x, dtype=torch.float)
print(x)

print(x.size())
"""

"""
x = torch.rand(5, 3)
y = torch.randn_like(x, dtype=torch.float)
z = torch.add(x, y)
print(z)

result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

# adds x to y
y.add_(x)
print(y)

print(x)
print(x[:,1])
"""

"""
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)
print(x)
print(y)
print(z)
print(x.size(), y.size(), z.size())
"""

"""
x = torch.randn(1)
print(x)
print(x.item())
"""

"""
# Tensor to Numpy
a = torch.ones(5)
b = a.numpy()
print(a, b)

a.add_(1)
print(a, b)
"""

"""
# Numpy to Tensor
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a, b)
"""

"""
if torch.cuda.is_available():
	device = torch.device("cuda")
	x = torch.randn(5, 3)
	y = torch.ones_like(x, device=device)
	x = x.to(device)
	z = x + y
	print(z)
	print(z.to("cpu", torch.double))
"""
