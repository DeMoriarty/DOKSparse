# Sparse DOK (Dictionary of Keys) Tensors on GPU
Common sparse matrix/tensor formats such as COO, CSR and LIL do not support constant time access/asignment of each individual tensor element. Because of that, PyTorch supports very limited indexing operations for its sparse tensor formats, and numpy-like advanced indexing is not supportd for the most part. 

**DOK (Dictionary of Keys)** is a sparse tensor format that uses a hashmap to store index-value pairs. Accessing any individual element, including elements that are zero, is theoretically constant time. DOK format can also be converted to uncoalesced COO format with minimal cost.

This repository contains an implementation of sparse DOK tensor format in CUDA and pytorch, as well as a hashmap as its backbone. The main goal of this project is to make sparse tensors behave as closely to dense tensors as possible.

**note:** currently only nvidia gpus are supported, contributions for cpu/rocm/metal support are welcomed!
## Installation
This package depends on [pytorch](https://pytorch.org/), [cupy](https://docs.cupy.dev/en/stable/install.html) and [sympy](https://docs.sympy.org/latest/install.html). Please make sure to have a newer versions of these packages before installing sparse_dok.
```bash
pip install sparse-dok
```

## Quick Start
### Sparse DOK Tensor
#### construction and conversion
there are various ways of creating a sparse DOK tensor:

1. construct with indices and values (similar to `torch.sparse_coo_tensor`):
```python
indices = torch.arange(100, device=device)[None].expand(2, -1)
values = torch.randn(100, device=device)

dok_tensor = SparseDOKTensor(size=(100, 100), indices=indices, values=values)
```

2. create an empty tensor first, set items later:
```python
import torch
from sparse_dok import SparseDOKTensor

device = "cuda:0"
dok_tensor = SparseDOKTensor(size=(100, 100), dtype=torch.float, device=device)

indices = torch.arange(100, device=device)

dok_tensor[indices, indices] = 1 # now this is a sparse identity matrix!
# assert torch.allclose(tensor.to_dense(), torch.eye(100, device=device))
```
3. convert from a dense tensor or sparse COO tensor:
```python
dok_tensor = SparseDOKTensor.from_dense(dense_tensor)
dok_tensor = SparseDOKTensor.from_sparse_coo(coo_tensor)
```

you can also convert a sparse DOK tensor to dense or sparse COO tensor:
```python
dense_tensor = dok_tensor.to_dense()
coo_tensor = dok_tensor.to_sparse_coo()
```

#### pytorch functions
sparse DOK tensors can be used in all pytorch functions that accept `torch.sparse_coo_tensor` as input, including some functions in `torch` and `torch.sparse`. In these cases, the sparse DOK tensor will be simply converted to `torch.sparse_coo_tensor` before entering the function.

```python
torch.add(dok_tensor, another_dok_tensor) # returns sparse coo tensor
torch.sparse.sum(dok_tensor, dim=0)
torch.sparse.mm(dok_tensor, dense_tensor)
...
```

Some `torch.Tensor` class methods are also implemented:

#### indexing, slicing and mutating
these methods are currently supported:
`select()`, `index_select()`, `__getitem__()`, `__setitem__()`, `transpose()`, `transpose_()`, `permute()`, `permute_()`, `T`, `t()`, `flatten()`, `reshape()`

**note**: `flatten()` and `reshape()` creates a copy of the original tensor, and rehashes all the index-value pairs, which makes it time consuming.

**note**: `transpose()`, `permute()`, `T` and `t()` return a view of the original tensor that shares the same storage.

**note**: `__getitem__()` and `__setitem__()` supports advanced slicing/indexing. for example:
```python
dok_tensor = SparseDOKTensor(size=(10,), values=values, indices=indices)

# indexing with integers
dok_tensor[0]

# indexing with lists/ndarrays/tensors of integers
dok_tensor[ [3, 6, 9] ]
# output shape: (3,)

dok_tensor[ np.arange(5) ]
# output shape: (5,)

dok_tensor[ torch.randint(10, size=(2, 2) ) ] 
# output shape: (2, 2)

# indexing with boolean mask
mask = torch.arange(10) < 5
dok_tensor[ mask ]
# output shape: (5,)

# slicing
dok_tensor[0:5]
# output shape: (5,)

# and any combination of the above
dok_tensor = SparseDOKTensor(size=(13, 11, 7, 3), values=values, indices=indices)
dok_tensor[5, :4, [3, 5, 7], torch.tensor([True, False, True])]
# output shape: (4, 3, 2)
```

**note**: `__getitem__()` always returns a dense tensor, similarly `__setitem__()` needs either a scalar value, or a broadcastable dense tensor as input. Sometimes slicing a large tensor may result in out of memory, so use it with caution.

#### some special operators
`sum()`, `softmax()`, `log_softmax()`, `norm()`, `normalize()`, `normalize_()`

**note**: `normalize()` is similar to `torch.nn.functional.normalize()`, and `normalize_()` is its inplace version.

#### other methods
`dtype`, `device`, `shape`, `ndim`, `sparsity`, `is_sparse`, `indices()`, `values()`, `_indices()`, `_values()`, `_nnz()`, `size()`, `clone()`, `resize()`,
`to_sparse_coo()`, `to_sparse_csr()`, `to_dense()`, `resize()`, `is_coalesced()`,
`abs()`, `sqrt()`, `ceil()`, `floor()`, `round()`

**note**: currently `torch.abs(dok_tensor)` returns sparse COO tensor, while `dok_tensor.abs()` returns sparse DOK tensor, same goes for all other unary functions. this behavior may change in the future.

### CUDA Hashmap
`CudaClosedHashmap` is a simple hashmap with closed hashing and linear probing. Both keys and values can be arbitrary shaped tensors \*. All keys must have the same shape and data type, same goes for all values. Get/set/remove operations are performed in batches, taking advantage of the GPUs parallel processing power, millions of operations can be performed within less than a fraction of a second. 

\* the number of elements each key and value can have is limited.

#### basic usage
```python
from sparse_dok import CudaClosedHashmap

hashmap = CudaClosedHashmap()

n_keys = 1_000_000
keys = torch.rand(n_keys, 3, device="cuda:0")
values = torch.randint(2**63-1, size=(n_keys, 5), device="cuda:0", dtype=torch.long)

### set items
hashmap[keys] = values
# or 
is_set = hashmap.set(keys, values)


### get items
result, is_found = hashmap[keys]
# or
result, is_found = hashmap.get(keys, fallback_value=0)

### remove items
del hashmap[keys]
# or 
is_removed = hashmap.remove(keys)
```

#### some details
##### 1. storage
`hashmap._keys` and `hashmap._values` are the tensors where the key-value pairs are stored, their shapes are `(num_buckets, *key_shape)` and `(num_buckets, *value_shape)`. for each unique key, a 64bit uuid is computed using a second hash function, and stored alongside with keys and values. the uuids are stored in`hashmap._uuid`, which has a shape of `(num_buckets, )`. 
`hashmap.keys()`, `hashmap.values()`, `hashmap.uuid()` filters out unoccupied buckets, and returns only the items that are stored by user.

`hashmap.n_buckets` is the current capacity of the hashmap.
the number of key-value pairs stored in the hashmap can be obtained from `hashmap.n_elements` or `len(hashmap)`

##### 2. automatic rehashing
Rehashing is triggered when the load factor (`n_elements / n_buckets`) of the hashmap reaches `rehash_threshold`. During a rehash, the capacity of the hashmap increases, and all the items will be rehashed with a different random hash function. The new number of buckets in the hashmap is equal to `n_elements * rehash_factor`. By default, `rehash_threshold = 0.75` and `rehash_factor = 2.0`.

To prevent frequent rehashing, you can set a higher initial capacity to the hashmap, or set `rehash_factor` to a higher value. increasing `rehash_threshold` is not recommended, because it may cause severe performance degradation.

```python
hashmap = CudaClosedHashmap(n_buckets=100000, rehash_factor=4.0)
```

You can also manually rehash a hashmap to a desired capacity:
```
hashmap.rehash(n_buckets=200000)
```

##### 3. nondeterministic behavior
To reduce global memory access latency of the cuda kernels, one optimization in `CudaClosedHashmap` is using uuid of keys to check whether two keys are identical, rather than comparing them directly. uuids are computed using a second hash function, and can have values ranging from $0$ to $2^{63} - 1$. It's technically possible for two keys to have the same uuid, however the chances are pretty small, and even if that happen, as long as they don't end up in the same bucket (determined by the first hash function), it will not cause any problem. Only when two different keys happen to have the exact same hashcodes from two different hash functions, one of them will be misidentified as the other.

##### 
