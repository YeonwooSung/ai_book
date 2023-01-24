# Reference: <https://towardsdatascience.com/useful-pytorch-functions-356de5f31a1e>

import torch

# 1. torch.linspace

# torch.linspace(start, end, steps=100, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
# Returns a one-dimensional tensor of size steps whose values are evenly spaced from start to end.
# The argument steps is a positive integer representing the number of samples to generate.
torch.linspace(1, 10)

'''
Output:
tensor([ 1.0000, 1.0909, 1.1818, 1.2727, 1.3636, 1.4545, 1.5455, 1.6364, 1.7273, 1.8182, 1.9091, 2.0000, 2.0909, 2.1818, 2.2727, 2.3636, 2.4545, 2.5455, 2.6364, 2.7273, 2.8182, 2.9091, 3.0000, 3.0909, 3.1818, 3.2727, 3.3636, 3.4545, 3.5455, 3.6364, 3.7273, 3.8182, 3.9091, 4.0000, 4.0909, 4.1818, 4.2727, 4.3636, 4.4545, 4.5455, 4.6364, 4.7273, 4.8182, 4.9091, 5.0000, 5.0909, 5.1818, 5.2727, 5.3636, 5.4545, 5.5455, 5.6364, 5.7273, 5.8182, 5.9091, 6.0000, 6.0909, 6.1818, 6.2727, 6.3636, 6.4545, 6.5455, 6.6364, 6.7273, 6.8182, 6.9091, 7.0000, 7.0909, 7.1818, 7.2727, 7.3636, 7.4545, 7.5455, 7.6364, 7.7273, 7.8182, 7.9091, 8.0000, 8.0909, 8.1818, 8.2727, 8.3636, 8.4545, 8.5455, 8.6364, 8.7273, 8.8182, 8.9091, 9.0000, 9.0909, 9.1818, 9.2727, 9.3636, 9.4545, 9.5455, 9.6364, 9.7273, 9.8182, 9.9091, 10.0000])
'''

torch.linspace(start=1, end=10, steps=5)

'''
Output:
tensor([ 1.0000,  3.2500,  5.5000,  7.7500, 10.0000])
'''


# 2. torch.eye

# torch.eye(n, m=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
# Returns a 2-D tensor with ones on the diagonal and zeros elsewhere.
# The shape of the tensor is (n, m) if m is given, else (n, n).

torch.eye(n=4, m=5)

'''
Output:
tensor([[1., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0.],
        [0., 0., 1., 0., 0.],
        [0., 0., 0., 1., 0.]])
'''

torch.eye(n=3)

'''
Output:
tensor([[1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]])
'''


# 3. torch.full

# torch.full(size, fill_value, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
# Returns a tensor filled with the scalar value fill_value, with the shape defined by the variable argument size.

torch.full(size=(2, 3), fill_value=5)

'''
Output:
tensor([[5, 5, 5],
        [5, 5, 5]])
'''


# 4. torch.cat

# torch.cat(tensors, dim=0, out=None) → Tensor
# Concatenates the given sequence of seq tensors in the given dimension.
# All tensors must either have the same shape (except in the concatenating dimension) or be empty.

x = torch.tensor([[1, 2, 3], [4, 5, 6]])
y = torch.tensor([[7, 8, 9], [10, 11, 12]])

torch.cat((x, y), dim=0)

'''
Output:
tensor([[ 1,  2,  3],
        [ 4,  5,  6],
        [ 7,  8,  9],
        [10, 11, 12]])
'''


# 5. torch.take

# torch.take(input, indices) → Tensor
# Returns a new tensor with the elements of input at the given indices.
# The input tensor is treated as if it were viewed as a 1-D tensor.

x = torch.tensor([[1, 2, 3], [4, 5, 6]])
torch.take(x, torch.tensor([0, 2, 5]))

'''
Output:
tensor([1, 3, 6])
'''


# 6. torch.unbind

# torch.unbind(tensor, dim=0) → tuple of Tensors
# Returns a tuple of all slices along a given dimension, already without it.

x = torch.tensor([[1, 2, 3], [4, 5, 6]])
torch.unbind(x, dim=0)

'''
Output:
(tensor([1, 2, 3]), tensor([4, 5, 6]))
'''



# 7. torch.Tensor.clone

# torch.Tensor.clone(input, *, memory_format=None) → Tensor
# torch.Tensor.clone returns a copy of the tensor with the same size and data type.
# When we create a copy of the tensor using x=y , changing one variable also affects the other variable since it points to the same memory location.

a = torch.tensor([[1., 2.],
                  [3., 4.],
                  [5., 6.]])
b = a
a[1,0]=9
b

'''
Output:
tensor([[1., 2.],
        [9., 4.],
        [5., 6.]])
'''
