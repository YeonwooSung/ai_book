# torchdim

## Named Tensors using First-class Dimensions in PyTorch

An implementation of named tensors with the functionality of einsum , batching (vmap, xmap), and tensor indexing by adding dimension objects to PyTorch.

The tensor input to a resnet might have the shape [8, 3, 224, 224] but informally we think of those dimensions as 'batch', 'channel', 'width', and 'height'. Eventhough 'width' and 'height' have the same size we still think of them as separate dimensions, and if we have two different images, we think of both as sharing the same 'channel' dimension.

Instead using string as a dimension name, the torchdim uses a python object called Dim.

```python
from torchdim import dims

# einsum
def mm(A: torch.Tensor, B: torch.Tensor):
    i, j, k = dims(3)
    r = (A[i, k] * B[k, j]).sum(k)
    return r.order(i, j)

# rearrange
def pixel_shuffle(img: torch.Tensor, upscale_factor=2):
    h2, w2, c, b, h, w = dims(6)
    h2.size = w2.size = upscale_factor
    return img[b, (c, h2, w2), h, w].order(b, c, (h, h2), (w, w2))

# batching
def bmm(A: torch.Tensor, B: torch.Tensor):
    i = dims(1)
    return mm(A[i], B[i]).order(i)

# indexing
def embedding_bag(input: torch.Tensor, embedding_weights: torch.Tensor):
    batch, sequence, features = dims(3)
    r = embedding_weights[input[batch, sequence], features].sum(sequence)
    return r.order(batch, features)
```
