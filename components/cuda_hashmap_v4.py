import uuid
from regex import E
import torch
import cupy as cp
import torch.nn as nn
import numpy as np
import math
import sympy
import random
from typing import Optional, Union, Iterable
from torchtimer import ProfilingTimer
from pathlib import Path
from torch import Tensor

from UWOT.components.cuda.hash_map_set_v4 import HashMapSetCuda
from UWOT.components.cuda.hash_map_get import HashMapGetCuda
from UWOT.util import str2dtype, next_power_of_2, unique_first_occurrence, expand_tensor, batch_allclose



class CudaHashMap:
  _hash_map_set_cuda = HashMapSetCuda(tpb=256, stages=4)
  _hash_map_get_cuda = HashMapGetCuda(tpb=256, stages=4)
  def __init__(
      self, 
      n_buckets : int,
      device : Union[str, torch.device] = "cuda:0",
      primes1 : Optional[Iterable] = None,
      primes2 : Optional[Iterable] = None,
      alpha1 : Optional[Iterable] = None,
      alpha2 : Optional[Iterable] = None,
      beta1 : Optional[Iterable] = None,
      beta2 : Optional[Iterable] = None,
      key_size : int = None,
      subkey_inds : Optional[Tensor] = None,
      rehash_factor : float = 2.0,
    ):
    self.n_buckets = n_buckets
    self._head_node = torch.zeros(n_buckets, device=device, dtype=torch.long) - 1
    self._next_node = torch.zeros(n_buckets, device=device, dtype=torch.long) - 1
    self._uuid = torch.zeros(n_buckets, device=device, dtype=torch.long) - 1
    # self.head_node_is_empty = torch.zeros(n_buckets, dtype=torch.bool, device=device)
    # self.head_node_is_empty.fill_(True)

    self._key_size = key_size
    self._keys = None
    self._values = None

    self._primes1 = primes1
    self._primes2 = primes2

    self._alpha1 = alpha1
    self._alpha2 = alpha2
    self._beta1 = beta1
    self._beta2 = beta2
    self._n_elements = 0
    self._n_nodes = n_buckets  # all head nodes + new nodes
    self._rehash_factor = rehash_factor
    self.device = torch.device(device)

    self._unique_subkeys = None
    self._subkey_inds = None
    if subkey_inds is not None:
      self._subkey_inds = subkey_inds
      self._unique_subkeys = CudaHashMap(32, device=self.device)


    self.timer = ProfilingTimer(name="CudaHashMap", enabled=False)
    self.debug = False

  @property
  def primes1(self):
    if self._primes1 is None:
      assert self._keys is not None or self._key_size is not None
      if self._keys is not None:
        key_size = self._keys[0].flatten().shape[0]
      elif self._key_size is not None:
        key_size = self._key_size
        
      self._primes1 = [np.random.randint(self.n_buckets, self.n_buckets*2) for i in range(key_size)]
      self._primes1 = torch.tensor([sympy.nextprime(i) for i in self._primes1], device=self.device)
    return self._primes1

  @property
  def primes2(self):
    if self._primes2 is None:
      assert self._keys is not None or self._key_size is not None
      if self._keys is not None:
        key_size = self._keys[0].flatten().shape[0]
      elif self._key_size is not None:
        key_size = self._key_size
      self._primes2 = [np.random.randint(0, 2**60, dtype="int64") for i in range(key_size)]
      self._primes2 = torch.tensor([sympy.nextprime(i) for i in self._primes2], device=self.device)
    return self._primes2

  @property
  def alpha1(self):
    if self._alpha1 is None:
      self._alpha1 = torch.tensor([np.random.randint(i.cpu()) - 1 for i in self.primes1], device=self.device)#torch.randint(0, size=self.primes1.shape)
    return self._alpha1
  
  @property
  def alpha2(self):
    if self._alpha2 is None:
      self._alpha2 = torch.tensor([np.random.randint(i.cpu()) - 1 for i in self.primes1], device=self.device)#torch.randint(0, size=self.primes1.shape)
    return self._alpha2
  
  @property
  def beta1(self):
    if self._beta1 is None:
      self._beta1 = torch.tensor([np.random.randint(i.cpu()) - 1 for i in self.primes1], device=self.device)#torch.randint(0, size=self.primes1.shape)
    return self._beta1
  
  @property
  def beta2(self):
    if self._beta2 is None:
      self._beta2 = torch.tensor([np.random.randint(i.cpu()) - 1 for i in self.primes1], device=self.device)#torch.randint(0, size=self.primes1.shape)
    return self._beta2

  @property
  def capacity(self):
    return self._uuid.shape[0]

  @property
  def keys(self):
    if self._keys is None:
      return None
    else:
      return self._keys[: self.n_elements ]

  @property
  def values(self):
    if self._values is None:
      return None
    else:
      return self._values[: self.n_elements ]

  @property
  def next_node(self):
    return self._next_node[:self.n_elements]
  
  @property
  def head_node(self):
    return self._head_node

  @property
  def uuid(self):
    return self._uuid[: self.n_elements ]

  @property
  def n_nodes(self):
    assert self._n_nodes >= self.n_buckets
    return self._n_nodes

  @property
  def n_elements(self):
    return self._n_elements

  @property
  def load_factor(self):
    return self.n_elements / self.n_buckets

  @property
  def key_shape(self):
    assert self.keys is not None
    return self.keys.shape[1:]
  
  @property
  def value_shape(self):
    assert self.values is not None
    return self.values.shape[1:]

  def expand(self, by=1.0):
    if by > 0:
      # print(f"expanding by {by}...")
      self._next_node = expand_tensor(self._next_node, fill=-1, by=by)
      self._uuid = expand_tensor(self._uuid, fill=-1, by=by)
      if self._keys is not None:
        self._keys = expand_tensor(self._keys, by=by)
      if self._values is not None:
        self._values = expand_tensor(self._values, by=by)
    
  def test_uniformity(self, keys):
    hash_codes = self.get_hash(keys)
    logits = (torch.bincount(hash_codes, minlength=self.n_buckets) + 1).log()
    def entropy(logits):
      logp = torch.log_softmax(logits, dim=-1)
      p = torch.softmax(logits, dim=-1)
      return -(logp * p).sum(dim=-1)
    return entropy(logits).exp() / self.n_buckets

  def __len__(self):
    return self._n_elements

  def get_hash(self, keys): #[n_keys, d_key]):
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
    if self._subkey_inds is None:
      h = ( (self.alpha1[None] * keys + self.beta1[None]) % self.primes1[None] ).prod(dim=-1).abs() % self.n_buckets
    else:
      h = ( (self.alpha1[None] * keys + self.beta1[None]) % self.primes1[None] )[:, self._subkey_inds].prod(dim=-1).abs() % self.n_buckets
    return h

  def get_uuid(self, keys): #[n_keys, d_key]):
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
    
    h = ( (self.alpha2[None] * keys + self.beta2[None]) % self.primes2[None] ).prod(dim=-1).abs()
    return h

  def set(self, new_keys, new_values):
    start, stop = self.timer.create_context("set")
    start("prep")
    assert new_keys.shape[0] == new_values.shape[0]
    if new_keys.shape[0] == 0:
      return

    if self._keys is None:
      self._keys = torch.empty(
        self.capacity,
        *new_keys.shape[1:],
        device=new_keys.device,
        dtype=new_keys.dtype
      )
      self._values = torch.empty(
        self.capacity,
        *new_values.shape[1:],
        device=new_values.device,
        dtype=new_values.dtype,
      )
    else:
      assert new_keys.dtype == self.keys.dtype
      assert new_keys.device == self.keys.device
      assert new_keys.shape[1:] == self.keys.shape[1:]
      assert new_values.dtype == self.values.dtype
      assert new_values.device == self.values.device
      assert new_values.shape[1:] == self.values.shape[1:]
    stop("prep")
    
    ### get uuid
    start("get uuid")
    new_uuid = self.get_uuid(new_keys) 
    stop("get uuid")

    # start("deduplicate")
    # ### deduplicate keys
    # uuid_is_unique = unique_first_occurrence(new_uuid)
    # new_keys = new_keys[uuid_is_unique]
    # new_values = new_values[uuid_is_unique]
    # new_uuid = new_uuid[uuid_is_unique]
    # stop("deduplicate")
    
    ### get hash code
    start("get hash code")
    new_hash_code = self.get_hash(new_keys)
    stop("get hash code")

    if self._unique_subkeys is not None:
      subkeys = new_keys.view(new_keys.shape[0], -1)[:, self._subkey_inds]
      self._unique_subkeys[subkeys] = torch.ones(new_keys.shape[0], device = self.device, dtype=torch.int32)

    start("p1")
    last_visited_node, status = CudaHashMap._hash_map_set_cuda.call_p1(
      new_hash_code = new_hash_code,
      new_uuid = new_uuid,
      uuid = self.uuid,
      next_node = self.next_node,
      head_node = self.head_node,
    )
    stop("p1")
    

    start("set value for found keys")
    is_key_found = status == 1
    self.values[last_visited_node[is_key_found]] = new_values[is_key_found]
    stop("set value for found keys")

    start("n_new")
    store_in_new_node = status != 1

    n_new = store_in_new_node.sum().item()
    stop("n_new")
    if n_new == 0:
      return

    start("expand if necessary")
    new_node = torch.arange(n_new, device=self.device) + self.n_elements

    expand_by = next_power_of_2(math.ceil((self.n_elements + n_new) / self.capacity)) - 1.0
    self.expand(by=expand_by)
    stop("expand if necessary")
    
    start("store keys in new node")
    ### increment number of elements stored in hash table 
    self._n_elements += n_new
    ### increment number of nodes in hash table 
    self._n_nodes += n_new

    store_in_new_node = torch.nonzero(store_in_new_node)[:, 0]
    new_hash_code = new_hash_code[store_in_new_node]
    new_hash_code, mapping = new_hash_code.sort(dim=0)
    store_in_new_node = store_in_new_node[mapping]

    last_visited_node = last_visited_node[store_in_new_node]    
    self.keys[new_node] = new_keys[store_in_new_node]
    self.values[new_node] = new_values[store_in_new_node]
    self.uuid[new_node] = new_uuid[store_in_new_node]
    stop("store keys in new node")
    start("p2")
    CudaHashMap._hash_map_set_cuda.call_p2(
      new_node=new_node,
      hash_code=new_hash_code,
      last_visited_node=last_visited_node,
      next_node=self.next_node,
      head_node=self.head_node,
    )
    if self._subkey_inds is None:
      if self.load_factor > 1.0:
        # self.rehash(next_power_of_2(self.n_elements)*3)
        self.rehash(int(self.n_elements * self._rehash_factor))
    else:
      if self._unique_subkeys is not None:
        if self._unique_subkeys.load_factor > 1.0:
          self.rehash(int(self._unique_subkeys.n_elements * self._rehash_factor))
      else:
        all_subkeys = self.keys.reshape(self.n_elements, -1)[:, self._subkey_inds]
        unique_subkeys = all_subkeys.unique(dim=0)
        load_factor = unique_subkeys.shape[0] / self.n_buckets
        if load_factor > 1.0:
          self.rehash(int(unique_subkeys.shape[0] * self._rehash_factor))

    stop("p2")

  def get(
      self,
      keys : torch.Tensor,
      default_value : Optional[torch.Tensor] = None,
    ):
    keys = keys.clone()
    if self.keys is None or self.values is None:
      return None, None

    if keys.shape[0] == 0:
      return torch.empty(0, *self.value_shape, device=self.device, dtype=self.values.dtype), torch.empty(0, device=self.device, dtype=torch.bool)

    assert keys.dtype == self.keys.dtype
    assert keys.device == self.keys.device
    assert keys.shape[1:] == self.key_shape
    if default_value is not None:
      if not isinstance(default_value, torch.Tensor):
        default_value = torch.tensor(default_value, device=self.device, dtype=self.values.dtype)
      default_value = torch.broadcast_to(default_value, self.value_shape)

    hash_code = self.get_hash(keys) #[n]
    uuid = self.get_uuid(keys)

    indices, is_key_found = CudaHashMap._hash_map_get_cuda(
      query_hash_code=hash_code, 
      query_uuid=uuid,
      uuid=self.uuid, 
      next_node=self.next_node,
      head_node=self.head_node,
    )
    values = self.values[indices]
    not_found = ~is_key_found
    if default_value is not None:
      values[not_found] = default_value[None]
    return values, not_found

  def rehash(self, n_buckets):
    # print(f"rehashing {n_buckets}")
    keys = self.keys.contiguous()
    values = self.values.contiguous()
    
    timer = self.timer
    self.__init__(
      n_buckets=n_buckets,
      key_size=self._key_size,
      device=self.device,
    )
    self.timer = timer
    self.set(keys, values)

  def __getitem__(self, keys):
    return self.get(keys)

  def __setitem__(self, keys, values):
    return self.set(keys, values)