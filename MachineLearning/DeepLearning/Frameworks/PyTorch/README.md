# PyTorch

[PyTorch](https://pytorch.org/) is an open source machine learning framework that accelerates the path from research prototyping to production deployment.

## Features

## Related libraries

1. [torchvision](./torchvision/)
2. [torchdim](./torchdim/)

## Discussions

### CUDA

#### Using cuda() method with a list type variable

If your data is a list type variable, then you cannot use the cuda() method directly. This is because that the list type is a built-in type which does not have a cuda method. The problem with your second approach is, that torch.nn.ModuleList is designed to properly handle the registration of torch.nn.Module components and thus does not allow passing tensors to it.

There are two ways to overcome this:

1. You could call .cuda on each element independently like this:

```python
data = [_data.cuda() for _data in data]
label = [_label.cuda() for _label in label] 
```

2. You could store your data elements in a large tensor (e.g. via torch.cat) and then call .cuda() on the whole tensor:

```python
data = []
label = []
    
for D, true_label in batch_dataLabel:
    D = D.float()
    true_label = true_label.float()
    # add new dimension to tensors and append to list
    data.append(D.unsqueeze(0))
    label.append(true_label.unsqueeze(0))

data = torch.cat(data, dim=0)
label = torch.cat(label, dim=0)

if gpu:
        data = data.cuda()
        label = label.cuda()
```

## Storage

Why does PyTorch force to use torch tensor for calculation? One of the main reason is that the PyTorch is using an unique concept call "storage". Basically, it uses a less bytes for each number than the python objects (python allocates at least 8 bytes to the number type object where c uses 4 bytes for int type and 2 bytes for short type). As the neural network contains a huge number of number type objects, it would be better to allocate as less memory as possible so that we do not waste the memory space unnecessarily.

When we make a tensor with PyTorch, the PyTorch allocates numbers in the data structure called "storage". It is a kind of list, but has some kind of metadata for tensor computations. By using the storage, you could make multiple tensors to share the same storage. You could also make different length of tensors with a single storage.

For example, if a storage contains [3, 1, 3, 4, 2, 4], and there are 2 tensors; a = [3, 1, 3, 4, 2, 4], b = [1, 3, 4, 2, 4]. In this case, the PyTorch links tensor a and b with storage, and just set the starting index of each tensor as corresponding value (a's starting index = 0, b's starting index = 1). In PyTorch, this starting index is called as a "offset".

Also, you might wonder how the PyTorch organises the n-th dimentional tensors. In this case, the PyTorch uses the thing called "stride". The stride lets the PyTorch know how much index they should move to get the next element.

The most beneficial thing that we could get by using PyTorch stride is that the PyTorch could perform the "transpose" operation easily by simply updating the stride object. By using stride, PyTorch does not need to re-allocate memory for the transposed tensor, it just need to transpose the "stride", which makes the tensor operation much faster.

Below is a sample code of checking storage, offset, and stride of the torch tensors.

```python

a = torch.tensor([[3,1,8], 
                  [0,9,2]])
print("============storage==========")
print(a.storage())
print("=============================")
print("Offset: ", a.storage_offset())
print("Strides: ", a.stride())

# Output:
# ============storage==========
#  3
#  1
#  8
#  0
#  9
#  2
# [torch.LongStorage of size 6]
# =============================
# Offset:  0
# Strides:  (3, 1)
```

```python

b = torch.tensor([[3,1],
                  [8,0],
                  [9,2]])
print("============storage==========")
print(b.storage())
print("=============================")
print("Offset: ", b.storage_offset())
print("Strides: ", b.stride())

# Output:
# ============storage==========
#  3
#  1
#  8
#  0
#  9
#  2
# [torch.LongStorage of size 6]
# =============================
# Offset:  0
# Strides:  (2, 1)
```

```python
a = torch.tensor([[3,1,1,2],
                  [8,0,3,4],
                  [9,2,5,6]])
print("============storage==========")
print(a.storage())
print("=============================")
print("Offset: ", a.storage_offset())
print("Strides: ", a.stride())

# Output:
# ============storage==========
#  3
#  1
#  1
#  2
#  8
#  0
#  3
#  4
#  9
#  2
#  5
#  6
# [torch.LongStorage of size 12]
# =============================
# Offset:  0
# Strides:  (4, 1)

b = a[1:3,1:3]
print("============storage==========")
print(b.storage())
print("=============================")
print("Offset: ", b.storage_offset())
print("Strides: ", b.stride())

# Output:
# ============storage==========
#  3
#  1
#  1
#  2
#  8
#  0
#  3
#  4
#  9
#  2
#  5
#  6
# [torch.LongStorage of size 12]
# =============================
# Offset:  5
# Strides:  (4, 1)

print(id(a.storage()), id(b.storage()))
# Output:
# 1447428900552 1447428900552
```

```python
a = torch.tensor([[3,1,1,2],
                  [8,0,3,4],
                  [9,2,5,6]])

b = a.t()
print("============storage==========")
print(b.storage())
print("=============================")
print("Offset: ", b.storage_offset())
print("Strides: ", b.stride())

# Output:
# ============storage==========
#  3
#  1
#  1
#  2
#  8
#  0
#  3
#  4
#  9
#  2
#  5
#  6
# [torch.LongStorage of size 12]
# =============================
# Offset:  0
# Strides:  (1, 4)
```
