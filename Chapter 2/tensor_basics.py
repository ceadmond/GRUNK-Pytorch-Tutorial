import torch
import numpy as np
from torch.autograd.function import Function

# print(torch.__version__)

"""
y = torch.rand(2, 3, 4, 5)
print(y.size())
"""

"""
a = torch.randn((3, 2))
numpy_a = a.numpy()
print(numpy_a)

torch_a = torch.from_numpy(numpy_a)
print(torch_a)
"""

"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
"""

"""
x = torch.rand(5, 5, requires_grad=True)
y = torch.rand(5, 5, requires_grad=True)
z = torch.sum(x + y)
z.backward()
print(x.grad, y.grad)
"""

"""
x = torch.rand(5, 5, requires_grad=True)
y = torch.rand(5, 5, requires_grad=True)
z = pow(x, 2) + pow(y, 3)

z.backward(torch.ones_like(x))
print(x.grad, y.grad)
"""

class MulConstant(Function):
	def forward(ctx, tensor, constant):
		ctx.constant = constant
		return tensor * constant

	def backward(ctx, grad_output):
		return grad_output, Nan

		
a = torch.rand(3, 3, requires_grad=True)
b = MulConstant.apply(a, 5)
print('a:'+str(a))
print('b:'+str(b))
