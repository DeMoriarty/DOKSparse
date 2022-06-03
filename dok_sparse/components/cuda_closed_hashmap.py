import torch
import numpy as np
import math
import sympy
from typing import Optional, Union, Iterable
from torchtimer import ProfilingTimer
from pathlib import Path
from torch import Tensor

from .cuda.closed_hashmap_impl import ClosedHashmapImplCuda
from ..util import str2dtype, next_power_of_2, unique_first_occurrence, expand_tensor, batch_allclose

class CudaClosedHashmap:
  def __init__(
      self, 
      n_buckets : int,
      device : Union[str, torch.device, int] = "cuda:0",
      prime1 : Optional[Iterable] = None,
      prime2 : Optional[Iterable] = None,
      alpha1 : Optional[Iterable] = None,
      alpha2 : Optional[Iterable] = None,
      beta1 : Optional[Iterable] = None,
      beta2 : Optional[Iterable] = None,
      key_size : int = None,
      key_perm : Optional[Tensor] = None,
      rehash_factor : float = 2.0,
      rehash_threshold : float = 0.75,
    ):
    self.n_buckets = n_buckets
    self.device = torch.device(device)
    self._uuid = torch.zeros(n_buckets, device=device, dtype=torch.long) - 1

    self._key_size = key_size
    self._keys = None
    self._values = None

    self._prime1 = prime1
    self._prime2 = prime2

    self._alpha1 = alpha1
    self._alpha2 = alpha2
    self._beta1 = beta1
    self._beta2 = beta2
    self._n_elements = torch.zeros(size=(), device=self.device, dtype=torch.long)
    self._rehash_factor = rehash_factor
    self._rehash_threshold = rehash_threshold

    self._unique_subkeys = None
    self._key_perm = None
    if key_perm is not None:
      if not isinstance(key_perm, Tensor):
        key_perm = torch.tensor(key_perm, dtype=torch.long, device=self.device)
      self._key_perm = key_perm.to(device=self.device, dtype=torch.long)
      # self._unique_subkeys = CudaClosedHashmap(32, device=self.device)

    self._hashmap_impl_cuda = None

    self.timer = ProfilingTimer(name="CudaClosedHashmap", enabled=False)
    self.debug = False

  @property
  def hashmap_impl_cuda(self):
    if self._hashmap_impl_cuda is None:
      assert self._keys is not None
      self._hashmap_impl_cuda = ClosedHashmapImplCuda(
        tpb=256,
        kpt=1,
        key_size=np.product(*self.key_shape),
        value_size=np.product(*self.value_shape),
        key_type=self._keys.dtype,
        value_type=self._values.dtype,
      )
    return self._hashmap_impl_cuda

  @property
  def prime1(self):
    if self._prime1 is None:
      assert self._keys is not None or self._key_size is not None
      if self._keys is not None:
        key_size = self._keys[0].flatten().shape[0]
      elif self._key_size is not None:
        key_size = self._key_size
        
      self._prime1 = [np.random.randint(self.n_buckets, self.n_buckets*2) for i in range(key_size)]
      self._prime1 = torch.tensor([sympy.nextprime(i) for i in self._prime1], device=self.device)
    return self._prime1

  @property
  def prime2(self):
    if self._prime2 is None:
      assert self._keys is not None or self._key_size is not None
      if self._keys is not None:
        key_size = self._keys[0].flatten().shape[0]
      elif self._key_size is not None:
        key_size = self._key_size
      self._prime2 = [np.random.randint(0, 2**60, dtype="int64") for i in range(key_size)]
      self._prime2 = torch.tensor([sympy.nextprime(i) for i in self._prime2], device=self.device)
    return self._prime2

  @property
  def alpha1(self):
    if self._alpha1 is None:
      self._alpha1 = torch.tensor([np.random.randint(i.cpu()) - 1 for i in self.prime1], device=self.device)#torch.randint(0, size=self.prime1.shape)
    return self._alpha1
  
  @property
  def alpha2(self):
    if self._alpha2 is None:
      self._alpha2 = torch.tensor([np.random.randint(i.cpu()) - 1 for i in self.prime1], device=self.device)#torch.randint(0, size=self.prime1.shape)
    return self._alpha2
  
  @property
  def beta1(self):
    if self._beta1 is None:
      self._beta1 = torch.tensor([np.random.randint(i.cpu()) - 1 for i in self.prime1], device=self.device)#torch.randint(0, size=self.prime1.shape)
    return self._beta1
  
  @property
  def beta2(self):
    if self._beta2 is None:
      self._beta2 = torch.tensor([np.random.randint(i.cpu()) - 1 for i in self.prime1], device=self.device)#torch.randint(0, size=self.prime1.shape)
    return self._beta2

  @property
  def key_perm(self):
    if self._key_perm is None:
      assert self._keys is not None
      self._key_perm = torch.arange(math.prod(self.key_shape), device=self.device)
    else:
      assert self._key_perm.shape[0] == math.prod(self.key_shape)
    return self._key_perm

  @key_perm.setter
  def key_perm(self, new):
    if not isinstance(new, Tensor):
      new = torch.tensor(new, device=self.device, dtype=torch.long)
      assert new.shape == self.key_perm.shape
      self._key_perm = new

  def keys(self) -> Optional[Tensor]:
    if self._keys is None:
      return None
    else:
      mask = self._uuid != -1
      keys = self._keys.reshape(self._keys.shape[0], -1)[mask][:, self.key_perm]
      return keys

  def values(self) -> Optional[Tensor]:
    if self._values is None:
      return None
    else:
      mask = self._uuid != -1
      return self._values[mask]

  def set_values(self, selectors, new_values):
    mask = self._uuid != -1
    masked_values = self._values[mask]
    masked_values[selectors] = new_values
    self._values[mask] = masked_values

  def permute_keys(self, ordering : tuple[int]):
    assert self._keys is not None, "hashmap is empty"
    assert len(self.key_shape) == 1, "keys are not one dimensional"
    assert len(ordering) == self.key_shape[0]
    assert len(set(ordering)) == len(ordering), "duplicates are not allowed"
    self._keys = self._keys[:, ordering]
    self._alpha1 = self._alpha1[ordering] 
    self._alpha2 = self._alpha2[ordering] 
    self._beta1 = self._beta1[ordering] 
    self._beta2 = self._beta2[ordering] 
    self._prime1 = self._prime1[ordering] 
    self._prime2 = self._prime2[ordering]
    self._key_size = math.prod(self._keys.shape[1:])

  def clone(self, deepclone=False):
    if deepclone:
      new = CudaClosedHashmap(
        n_buckets=self.n_buckets, 
        device=self.device,
        prime1=self.prime1.clone(),  
        prime2=self.prime2.clone(), 
        alpha1=self.alpha1.clone(), 
        alpha2=self.alpha2.clone(), 
        beta1=self.beta1.clone(), 
        beta2=self.beta2.clone(),
        key_perm=self._key_perm.clone(),
        rehash_factor=self._rehash_factor,
        rehash_threshold=self._rehash_threshold,
      )
      new._keys = self._keys.clone()
      new._values = self._values.clone()
      new._uuid = self._uuid.clone()
      new._n_elements = self._n_elements.clone()
      new._hashmap_impl_cuda = self._hashmap_impl_cuda
      return new
    else:
      new = CudaClosedHashmap(
        n_buckets=self.n_buckets, 
        device=self.device,
        prime1=self.prime1,  
        prime2=self.prime2, 
        alpha1=self.alpha1, 
        alpha2=self.alpha2, 
        beta1=self.beta1, 
        beta2=self.beta2,
        key_perm=self._key_perm,
        rehash_factor=self._rehash_factor,
        rehash_threshold=self._rehash_threshold,
      )
      new._keys = self._keys
      new._values = self._values
      new._uuid = self._uuid
      new._n_elements = self._n_elements
      new._hashmap_impl_cuda = self._hashmap_impl_cuda
      return new

  def uuid(self) -> Tensor:
    mask = self._uuid != -1
    return self._uuid[mask]

  @property
  def n_elements(self) -> int:
    return self._n_elements.item()

  @property
  def load_factor(self) -> float:
    return self.n_elements / self.n_buckets

  @property
  def key_shape(self) -> torch.Size:
    assert self._keys is not None
    return self._keys.shape[1:]
  
  @property
  def value_shape(self) -> torch.Size:
    assert self._values is not None
    return self._values.shape[1:]
    
  def test_uniformity(self, keys):
    hash_codes = self.get_hash(keys)
    logits = (torch.bincount(hash_codes, minlength=self.n_buckets) + 1).log()
    def entropy(logits):
      logp = torch.log_softmax(logits, dim=-1)
      p = torch.softmax(logits, dim=-1)
      return -(logp * p).sum(dim=-1)
    return entropy(logits).exp() / self.n_buckets

  def __len__(self) -> int:
    return self.n_elements

  def get_hash(self, keys : Tensor) -> Tensor: #[n_keys, d_key]):
    # torch.quantize_per_tensor(torch.randn(1000,8), 0.1, 0, torch.qint32).int_repr().sum(-1) % 10
    n_keys = keys.shape[0]
    keys = keys.reshape(n_keys, -1).clone()
    if keys.dtype == torch.float:
      keys = keys.view(dtype=torch.int32).long()
      ### swap -0.0 with 0.0
      keys[keys == -2**31] = 0
    elif keys.dtype == torch.half:
      keys = keys.view(dtype=torch.int16).long()
      keys[keys == -2**15] = 0
    elif keys.dtype == torch.double:
      keys = keys.view(dtype=torch.int64)
      keys[keys == -2**63] = 0
    else:
      keys = keys.to(torch.long)
    
    # modified from https://www.youtube.com/watch?v=Kf2V77ut-B0
    if self._key_perm is None:
      h = ( (self.alpha1[None] * keys + self.beta1[None]) % self.prime1[None] ).prod(dim=-1).abs() % self.n_buckets
    else:
      h = ( (self.alpha1[None] * keys + self.beta1[None]) % self.prime1[None] )[:, self._key_perm].prod(dim=-1).abs() % self.n_buckets
    return h

  def get_uuid(self, keys : Tensor) -> Tensor: #[n_keys, d_key]):
    n_keys = keys.shape[0]
    keys = keys.reshape(n_keys, -1).clone()
    if keys.dtype == torch.float:
      keys = keys.view(dtype=torch.int32).long()
      ### swap -0.0 with 0.0
      keys[keys == -2**31] = 0
    elif keys.dtype == torch.half:
      keys = keys.view(dtype=torch.int16).long()
      keys[keys == -2**15] = 0
    elif keys.dtype == torch.double:
      keys = keys.view(dtype=torch.int64)
      keys[keys == -2**63] = 0
    else:
      keys = keys.to(torch.long)
    
    h = ( (self.alpha2[None] * keys + self.beta2[None]) % self.prime2[None] ).prod(dim=-1).abs()
    return h

  def set(self, new_keys : Tensor, new_values : Tensor):
    start, stop = self.timer.create_context("set")
    start("prep")
    assert new_keys.shape[0] == new_values.shape[0]

    if self._keys is None:
      self._keys = torch.zeros(
        self.n_buckets,
        *new_keys.shape[1:],
        device=self.device,
        dtype=new_keys.dtype
      )
      self._values = torch.zeros(
        self.n_buckets,
        *new_values.shape[1:],
        device=self.device,
        dtype=new_values.dtype,
      )
    else:
      assert new_keys.dtype == self._keys.dtype
      assert new_keys.device == self._keys.device == self.device
      assert new_keys.shape[1:] == self._keys.shape[1:]
      assert new_values.dtype == self._values.dtype
      assert new_values.device == self._values.device == self.device
      assert new_values.shape[1:] == self._values.shape[1:]
    new_keys = new_keys.contiguous()
    new_values = new_values.contiguous()

    if new_keys.shape[0] == 0:
      return
    stop("prep")

    _, is_found = self.hashmap_impl_cuda.get(
      prime1=self.prime1,
      prime2=self.prime2,
      alpha1=self.alpha1,
      alpha2=self.alpha2,
      beta1=self.beta1,
      beta2=self.beta2,
      key_perm=self.key_perm,
      keys=new_keys,
      all_keys=self._keys,
      all_values=self._values,
      all_uuids=self._uuid
    )
    n_new_elements = new_keys[~is_found].unique(dim=0).shape[0]

    if self.n_elements + n_new_elements > int(self._rehash_threshold * self.n_buckets):
      self.rehash( int((self.n_elements + n_new_elements) * self._rehash_factor) )

    is_stored = self.hashmap_impl_cuda.set(
      prime1=self.prime1,
      prime2=self.prime2,
      alpha1=self.alpha1,
      alpha2=self.alpha2,
      beta1=self.beta1,
      beta2=self.beta2,
      key_perm=self.key_perm,
      keys=new_keys,
      values=new_values,
      all_keys=self._keys,
      all_values=self._values,
      all_uuids=self._uuid
    )
    # print( self._keys )
    # print(self._uuid)
    # print( (self._uuid == -1).sum(), self._uuid.shape)

    self._n_elements += n_new_elements

  def get(
      self,
      keys : torch.Tensor,
      fallback_value : Optional[torch.Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
    if self._keys is None or self._values is None:
      raise RuntimeError("store something first")
      return None, None

    if keys.shape[0] == 0:
      return torch.empty(0, *self.value_shape, device=self.device, dtype=self._values.dtype), torch.empty(0, device=self.device, dtype=torch.bool)
    assert keys.dtype == self._keys.dtype
    assert keys.device == self._keys.device
    assert keys.shape[1:] == self.key_shape
    if fallback_value is not None:
      if not isinstance(fallback_value, torch.Tensor):
        fallback_value = torch.tensor(fallback_value, device=self.device, dtype=self._values.dtype)
      fallback_value = torch.broadcast_to(fallback_value, self.value_shape)
    keys = keys.contiguous()
    values, is_found = self.hashmap_impl_cuda.get(
      prime1=self.prime1,
      prime2=self.prime2,
      alpha1=self.alpha1,
      alpha2=self.alpha2,
      beta1=self.beta1,
      beta2=self.beta2,
      key_perm=self.key_perm,
      keys=keys,
      all_keys=self._keys,
      all_values=self._values,
      all_uuids=self._uuid,
      fallback_value=fallback_value,
    )
    # print(values)
    # print(is_found.sum(), is_found.numel())
    return values, is_found

  def rehash(self, n_buckets):
    # print(f"rehashing {n_buckets}")
    keys = self.keys().contiguous()
    values = self.values().contiguous()
    
    timer = self.timer
    self.__init__(
      n_buckets=n_buckets,
      device=self.device,
      key_perm=self._key_perm,
      rehash_factor=self._rehash_factor,
      rehash_threshold=self._rehash_threshold,
    )
    self.timer = timer
    self.set(keys, values)

  def __getitem__(self, keys):
    return self.get(keys)

  def __setitem__(self, keys, values):
    return self.set(keys, values)