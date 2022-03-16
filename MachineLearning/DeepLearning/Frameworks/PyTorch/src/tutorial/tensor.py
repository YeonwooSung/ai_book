import torch



# scalars

scalar1 = torch.tensor([1.])
print(scalar1)

scalar2 = torch.tensor([3.])
print(scalar2)


# vectors

vector1 = torch.tensor([1., 2., 3.])
print(vector1)

vector2 = torch.tensor([4., 5., 6.])
print(vector2)

add_vector = vector1 + vector2
print(add_vector)
sub_vector = vector1 - vector2
print(sub_vector)
mul_vector = vector1 * vector2
print(mul_vector)
div_vector = vector1 / vector2
print(div_vector)

print(torch.add(vector1, vector2))
print(torch.sub(vector1, vector2))
print(torch.mul(vector1, vector2))
print(torch.div(vector1, vector2))
print(torch.dot(vector1, vector2))


# matrices

matrix1 = torch.tensor([[1., 2.], [3., 4.]])
matrix2 = torch.tensor([[5., 6.], [7., 8.]])

print(matrix1)
print(matrix2)

print(matrix1 + matrix2)
print(torch.add(matrix1, matrix2))
print(torch.sub(matrix1, matrix2))
print(torch.mul(matrix1, matrix2))
print(torch.div(matrix1, matrix2))
print(torch.matmul(matrix1, matrix2))


# tensor - matrix is a 2D array, but the tensor is ND array, where N > 0

tensor1 = torch.tensor([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]])
print(tensor1)

print(tensor1 + tensor1)
print(torch.add(tensor1, tensor1))
print(torch.sub(tensor1, tensor1))
print(torch.mul(tensor1, tensor1))
print(torch.div(tensor1, tensor1))
print(torch.matmul(tensor1, tensor1))
