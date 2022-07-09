# PyTorch

[PyTorch](https://pytorch.org/) is an open source machine learning framework that accelerates the path from research prototyping to production deployment.

## Features

## Related libraries

1. [torchvision](./torchvision/)

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
