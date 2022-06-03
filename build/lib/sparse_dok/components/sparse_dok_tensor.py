import torch
import numpy as np
from torch import Tensor

from .cuda_closed_hashmap import CudaClosedHashmap
from ..util import str2dtype, next_power_of_2
from .. import functions as fs

class SparseDOKTensor(object):
  def __init__(self, size, indices=None, values=None, device=None, dtype=None):
    assert isinstance(size, (list, tuple))
    self._size = torch.Size(size)
    
    if device is None:
      if indices is not None:
        self.device = indices.device
      else:
        self.device = "cuda:0"
    else:
      self.device = torch.device(device)

    if dtype is None:
      if values is not None:
        self._dtype = values.dtype
      else:
        self._dtype = torch.float32
    else:
      self._dtype = str2dtype(dtype)

    if indices is None:
      self._hashmap = CudaClosedHashmap(
        n_buckets=32,
        device=self.device,
        # subkey_inds=self.subkey_inds,
      )
    else:
      assert values is not None
      assert len(indices.shape) == 2
      assert len(values.shape) == 1
      assert indices.shape[1] == values.shape[0]
      self._hashmap = CudaClosedHashmap(
        n_buckets=next_power_of_2(values.shape[0]),
        device=self.device,
        # subkey_inds=self.subkey_inds,
      )
      self._hashmap.set(
        indices.T.to(self.device),
        values.to(device=self.device, dtype=self.dtype)[:, None]
      )

  @classmethod
  def from_sparse_coo(cls, tensor):
    assert isinstance(tensor, Tensor)
    assert tensor.layout == torch.sparse_coo
    tensor = tensor.coalesce()
    indices = tensor._indices()
    values = tensor._values()

    return cls(
      size=tensor.shape,
      indices=indices, 
      values=values, 
      device=tensor.device
    )

  @classmethod
  def from_dense(cls, tensor):
    assert isinstance(tensor, Tensor)
    assert tensor.layout == torch.strided
    return cls.from_sparse_coo(tensor.to_sparse())

  @classmethod
  def __torch_function__(cls, func, types, args=(), kwargs=None):
    if kwargs is None:
      kwargs = {}

    args = [arg.to_sparse_coo() if isinstance(arg, cls) else arg for arg in args]
    kwargs = {key: (arg.to_sparse_coo() if isinstance(arg, cls) else arg) for key, arg in kwargs.items()}

    return func(*args, **kwargs)
    
  @property
  def is_sparse(self):
    return True

  def _is_coalesced(self):
    return False

  def size(self):
    return self._size

  @property
  def shape(self):
    return self.size()

  @property
  def dtype(self):
    return self._dtype

  def clone(self, deepclone=False):
    new =  SparseDOKTensor(
      size=self._size, 
      indices=None, 
      values=None, 
      device=self.device, 
      dtype=self.dtype, 
    )
    new._hashmap = self._hashmap.clone(deepclone=deepclone)
    return new

  def indices(self):
    # assert self._hashmap.keys() is not None
    indices = self._hashmap.keys()
    if indices is None:
      return torch.empty(self.ndim, 0, device=self.device, dtype=torch.long)
    else:
      return indices.T

  def values(self):
    values = self._hashmap.values()
    if values is None:
      return torch.empty(0, device=self.device, dtype=self.dtype)
    else:
      return values.flatten()

  def set_values(self, selectors, new_values):
    self._hashmap.set_values(selectors=selectors, new_values=new_values[:, None])

  def _indices(self):
    return self._hashmap._keys

  def _values(self):
    return self._hashmap._values
  
  @property
  def ndim(self):
    return len(self.shape)

  def _nnz(self):
    return self._hashmap.n_elements

  @property
  def sparsity(self):
    return 1 - self._nnz() / float(np.prod(self.shape))

  @property
  def _as_sparse_coo_tensor(self):
    return self.to_sparse_coo()

  def to_sparse_coo(self): 
    t = torch.sparse_coo_tensor(
      indices=self.indices(), 
      values=self.values(),
      size=self.shape,
      device=self.device,
      )
    # t._coalesced_(True)
    t = t.coalesce()
    return t

  def to_sparse_csr(self):
    self.to_sparse_coo().to_sparse_csr()

  def to_dense(self):
    return self.to_sparse_coo().to_dense()
  
  def resize(self, *new_size):
    new_size = new_size + (self.ndim - len(new_size)) * (-1, )
    assert len(new_size) == self.ndim
    new_size = torch.Size([self.shape[i] if new_size[i] == -1 else new_size[i] for i in range(self.ndim)])
    self._size = new_size

  def is_coalesced(self):
    return self._is_coalesced

  def _handle_selectors(self, selectors):
    if isinstance(selectors, (Tensor, slice) ):
      selectors = (selectors, )

    _selectors = []
    for selector in selectors:
      if isinstance(selector, Tensor) and selector.dtype == torch.bool:
        _selectors += selector.nonzero(as_tuple=True)

      elif isinstance(selector, Tensor) and selector.dtype in {torch.int64, torch.int32, torch.int16}:
        _selectors.append(selector.long())
      
      elif isinstance(selector, np.ndarray) and selector.dtype == np.bool8:
        selector = torch.from_numpy(selector, device=self.device)
        _selectors.append(selector.nonzero(as_tuple=True))

      elif isinstance(selector, np.ndarray) and selector.dtype in {np.int16, np.int32, np.int64}:
        selector = torch.from_numpy(selector, device=self.device, dtype=torch.long)
        _selectors.append(selector)

      elif isinstance(selector, int):
        _selectors.append(torch.tensor(selector, device=self.device, dtype=torch.long))

      elif isinstance(selector, slice):
        _selectors.append(selector)

      elif isinstance(selector, (tuple, list)):
        selector = torch.tensor(selector, device=self.device, dtype=torch.long)
        _selectors.append(selector)

      else:
        raise ValueError
    selectors = _selectors + [slice(None)] * (self.ndim - len(_selectors))
    slices = [selector for selector in selectors if isinstance(selector, slice)]
    if len(selectors) != len(slices):
      broadcasted_shape = torch.broadcast_shapes(*[index.shape for index in selectors if isinstance(index, Tensor)])
    else:
      broadcasted_shape = tuple()

    selector_ndim = len(broadcasted_shape) + len(slices)
    # broadcasted_shape = torch.broadcast_shapes(*[index.shape for index in selectors if isinstance(index, Tensor)])
    try:
      first_tensor_idx = [type(i) for i in selectors].index(Tensor)
    except ValueError:
      first_tensor_idx = None

    idx = 0
    first_tensor_met = False
    for i in range(len(selectors)):
      if isinstance(selectors[i], Tensor):
        newaxis = [None] * selector_ndim
        for j in range(first_tensor_idx, first_tensor_idx + len(broadcasted_shape)):
          newaxis[j] = slice(None)
        selectors[i] = selectors[i][newaxis]
        if not first_tensor_met:
          idx += len(broadcasted_shape)
          first_tensor_met = True

      elif isinstance(selectors[i], slice):
        start, stop = selectors[i].start, selectors[i].stop
        start = 0 if start is None else start
        stop = self.shape[i] if stop is None else stop
        stop = max(min(stop, self.shape[i]), start)
        # stop = max(start, self.shape[i] if stop is None else stop)
        newaxis = [None] * selector_ndim
        newaxis[idx] = slice(None)
        selectors[i] = torch.arange(start, stop, device=self.device)[newaxis]
        idx += 1
    selectors = torch.stack(torch.broadcast_tensors(*selectors))
    return selectors

  def __getitem__(self, selectors):
    selectors = self._handle_selectors(selectors)
    values_shape = selectors.shape[1:]
    selectors = selectors.view(self.ndim, -1)

    if self._hashmap._values is None:
      values = torch.zeros(selectors.shape[1:], device=selectors.device, dtype=self.dtype)
    else:
      values, _ = self._hashmap.get(selectors.T, fallback_value=0)
    values = values.view(values_shape)
    return values

  def __setitem__(self, selectors, values):
    if not isinstance(values, Tensor):
      values = torch.tensor(values, device=self.device, dtype=self.dtype)
    else:
      assert values.dtype == self.dtype

    selectors = self._handle_selectors(selectors).unbind(dim=0)
    broadcasted = torch.broadcast_tensors(*selectors, values)
    selectors = torch.stack(broadcasted[:-1]).view(self.ndim, -1).T.contiguous()
    values = broadcasted[-1].view(-1)[:, None].contiguous()
    self._hashmap.set(selectors, values)

  def _get_slice_mask(self, slice, dim=0):
    start, stop, step = slice.start, slice.stop, slice.step
    if step is not None or step != 1:
      raise RuntimeError("only step size of 1 is supported")

    if start is None:
      start = 0

    if stop is None:
      stop = self.nnz

    if stop < start:
      stop = start

    indices = self.indices[dim]
    return (start <= indices) & (indices < stop)

  def _get_slice(self, slices):
    slices = tuple(slices) + (None,) * (self.ndim - len(slices))
    mask = torch.ones(self.nnz, device=self.device, dtype=torch.bool)
    new_size = []
    for dim, slice in enumerate(slices):
      if slice is None:
        new_size.append(self.shape[dim])
        continue
      new_size.append(max(slice.stop - slice.start, 0))
      mask = mask & self._get_slice_mask(slice, dim=dim)

    final_indices = self.indices[:, mask]
    final_values = self.values[mask]
    new_view = SparseDOKTensor(indices=final_indices, values=final_values, size=new_size, device=self.device, dtype=self.dtype)
    return new_view

  def norm(self, dim=None, p=2, keepdim=False):
    return fs.norm(
      input=self.to_sparse_coo(), 
      dim=dim, 
      p=p, 
      keepdim=keepdim
    )

  def sum(self, dim=None, keepdim=False):
    return fs.sum(
      input=self.to_sparse_coo(), 
      dim=dim, 
      keepdim=keepdim
    )

  def storage(self):
    return self._hashmap

  def normalize(self, dim = -1, p = 2, eps = 1e-12):
    return fs.normalize(
      input=self.to_sparse_coo(),
      dim=dim,
      p=p,
      eps=eps
    )
    
  def normalize_(self, dim = -1, p = 2, eps = 1e-12):
    return fs.normalize_(
      input=self.to_sparse_coo(),
      dim=dim,
      p=p,
      eps=eps
    )

  def softmax(self, dim=None):
    return fs.softmax(
      input=self.to_sparse_coo(), 
      dim=dim, 
    )

  def log_softmax(self, dim=None):
    return fs.log_softmax(
      input=self.to_sparse_coo(), 
      dim=dim, 
    )

  def index_select(self, dim, index):
    return self.to_sparse_coo().index_select(dim=dim, index=index)
  
  def select(self, dim, index):
    return self.to_sparse_coo().select(dim=dim, index=index)

  def permute_(self, *ordering):
    # self._hashmap.permute_keys(ordering)
    self._hashmap.key_perm = ordering
    self._size = torch.Size(self._size[i] for i in ordering)
    return self
  
  def permute(self, *ordering):
    new = self.clone()
    # new._hashmap.permute_keys(ordering)
    new._hashmap.key_perm = ordering
    new._size = torch.Size(new._size[i] for i in ordering)
    return new

  def transpose_(self, dim0, dim1):
    assert dim0 != dim1
    dim0 = dim0 % self.ndim
    dim1 = dim1 % self.ndim
    ordering = np.arange(self.ndim)
    ordering[dim0] = dim1
    ordering[dim1] = dim0
    return self.permute_(*ordering)

  def transpose(self, dim0, dim1):
    # return self.to_sparse_coo().transpose(dim0=dim0, dim1=dim1)
    assert dim0 != dim1
    dim0 = dim0 % self.ndim
    dim1 = dim1 % self.ndim
    ordering = np.arange(self.ndim)
    ordering[dim0] = dim1
    ordering[dim1] = dim0
    return self.permute(*ordering)

  def t(self):
    assert self.ndim == 2, "tensor isn't 2 dimensional, try x.transpose(dim1, dim2)"
    return self.transpose(0, 1)
  
  @property
  def T(self):
    return self.t()


