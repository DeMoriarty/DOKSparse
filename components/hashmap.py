import torch
import torch.nn as nn
import numpy as np
import math
import sympy
import random
from typing import Union

from torchtimer import ProfilingTimer

def unique_first_occurrence(x=None, unique=None, inverse=None, return_indices=False, dim=None):
  if unique is None and inverse is None:
    unique, inverse = torch.unique(x, dim=dim, sorted=True, return_inverse=True)
  elif x is None:
    unique, inverse = unique, inverse
  else:
    raise RuntimeError("must provide x or unique and inverse")

  perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
  inverse_flipped, perm = inverse.flip([0]), perm.flip([0])
  perm = inverse_flipped.new_empty(unique.size(0)).scatter_(0, inverse_flipped, perm)
  if return_indices:
    return perm
  else:
    mask = torch.zeros_like(inverse, dtype=torch.bool)
    mask[perm] = True    
    return mask 

class BaseHashMap:
  def __init__(
      self, 
      n_buckets : int,
      prime : int = None,
      alpha : int = None,
      beta : int = None,
      device : Union[str, torch.device] ="cpu"
    ):
    self.n_buckets = n_buckets
    self._next_node = torch.zeros(n_buckets, device=device, dtype=torch.long) - 1
    self._prev_node = torch.zeros(n_buckets, device=device, dtype=torch.long) - 1
    self.head_node_is_empty = torch.zeros(n_buckets, dtype=torch.bool, device=device)
    self.head_node_is_empty.fill_(True)

    self._keys = None
    self._values = None

    if prime is None:
      self._prime = sympy.nextprime(n_buckets)
    else:
      self._prime = prime

    if alpha is None:
      self._alpha = random.randint(0, self.prime - 1)
    else:
      self._alpha = alpha

    if beta is None:
      self._beta = random.randint(0, self.prime - 1)
    else:
      self._beta = beta

    # self._tensors = [self.next_node, self.prev_node]
    self._n_elements = 0
    self._n_nodes = n_buckets  # all head nodes + new nodes
    self.device = torch.device(device)

    self.timer = ProfilingTimer()

  @property
  def alpha(self):
    return self._alpha
  
  @property
  def beta(self):
    return self._beta
  
  @property
  def prime(self):
    return self._prime

  @property
  def capacity(self):
    return self._next_node.shape[0]

  @property
  def keys(self):
    if self._keys is None:
      return None
    else:
      return self._keys[: self.n_nodes ]

  @property
  def values(self):
    if self._values is None:
      return None
    else:
      return self._values[: self.n_nodes ]

  @property
  def next_node(self):
    return self._next_node[: self.n_nodes ]
  
  @property
  def prev_node(self):
    return self._prev_node[: self.n_nodes ]

  @property
  def n_nodes(self):
    assert self._n_nodes >= self.n_buckets
    return self._n_nodes

  @property
  def n_elements(self):
    return self._n_elements

  @staticmethod
  def _expand_tensor(tensor, dim=0, fill=0):
    shape = list(tensor.shape)
    n_dims = len(shape)
    assert n_dims > 0, "tensor needs to have at least 1 dimension"
    
    # shape[dim] *= 2
    new_tensor = torch.zeros(shape, dtype=tensor.dtype, device=tensor.device)
    new_tensor.fill_(fill)
    new_tensor = torch.cat([tensor, new_tensor], dim=dim)
    # slices = [slice(0, tensor.shape[dim]) if d == dim else slice(None) for d in range(n_dims)]
    # new_tensor[slices] = tensor
    return new_tensor

  def expand(self):
    self._next_node = self._expand_tensor(self._next_node, fill=-1)
    self._prev_node = self._expand_tensor(self._prev_node, fill=-1)
    if self._keys is not None:
      self._keys = self._expand_tensor(self._keys)
    if self._values is not None:
      self._values = self._expand_tensor(self._values)
    
  def __len__(self):
    return self._n_elements

  def get_hash(self, keys): #[n_keys, d_key]):
    # torch.quantize_per_tensor(torch.randn(1000,8), 0.1, 0, torch.qint32).int_repr().sum(-1) % 10
    assert len(keys.shape) == 2, "keys must be 2 dimentional tensors"
    n_keys, d_key = keys.shape
    if keys.dtype in {torch.float, torch.half, torch.double}:
      keys = torch.quantize_per_tensor(keys, 0.1, 0, torch.qint32).int_repr().long()
    else:
      keys = keys.to(torch.long)

    # modified from https://www.youtube.com/watch?v=Kf2V77ut-B0
    # h(c) = ((ac + b) mode p) mode m
    # h = ((self.alpha * keys + self.beta).sum(dim=-1) % self.prime) % self.n_buckets
    h = ((self.alpha * keys + self.beta).prod(dim=-1) % self.prime) % self.n_buckets
    # h = ((self.alpha * keys + self.beta) % self.prime).sum(dim=-1) % self.n_buckets
    # h = ((self.alpha * keys + self.beta) % self.prime).prod(dim=-1) % self.n_buckets
    return h

  @staticmethod
  def all_close(a, b, tol=1e-6):
    a, b = torch.broadcast_tensors(a, b)
    assert a.dtype == b.dtype
    assert a.device == b.device
    assert a.shape == b.shape
    assert len(a.shape) > 1
    batch_size = a.shape[0]
    a = a.view(batch_size, -1)
    b = b.view(batch_size, -1)
    if a.dtype in {torch.int, torch.long, torch.int16, torch.int8, torch.uint8, torch.bool}:
      return (a != b).sum(dim=-1) == 0

    elif a.dtype in {torch.half, torch.float, torch.double, torch.bfloat16}:
      return ((a - b).abs() < tol).sum(dim=-1) == 0

  def get_brtfrc(self, keys):
    if self._keys is None:
      return None, None

    batch_size = keys.shape[0]
    assert keys.shape[1:] == self.keys.shape[1:]

    values = torch.zeros(
      batch_size, 
      *self.values.shape[1:],
      device=self.values.device,
      dtype=self.values.dtype,
    )
    not_found = torch.empty(
      batch_size,
      device=values.device,
      dtype=torch.bool,
    )
    not_found.fill_(False)


    for i in range(batch_size):
      key = keys[i]
      mask = self.all_close(self.keys, key[None])
      if mask.sum() == 0:
        not_found[i] = True
        print("not found brtfrc")
        continue

      key_idx = torch.nonzero(mask)[0]
      values[i] = self.values[key_idx.unbind()]

    return values, not_found

  def get_loop(
      self, 
      keys : torch.Tensor,  # [n, *key_shape]
    ):
    keys = keys.clone()
    if self.keys is None:      
      return None, None

    assert keys.dtype == self.keys.dtype
    assert keys.device == self.keys.device
    assert keys.shape[1:] == self.keys.shape[1:]

    batch_size = keys.shape[0]
    values = torch.zeros(
      batch_size, 
      *self.values.shape[1:],
      device=self.values.device,
      dtype=self.values.dtype,
    ) - 11
    
    current_node = self.get_hash(keys) #[n]
    ### First, if the head node is empty, then set not_found to True
    not_found = self.head_node_is_empty[current_node]
    # print(f"{self.head_node_is_empty.sum()=}")

    for i in range(batch_size):
      if not_found[i].item():
        print("not found wtf head")
        continue
      c_node = current_node[i]
      c_key = keys[i]
      
      depth = 0
      key_list = []
      c_node_list = []
      max_iters = 1000
      while True:
      # for j in range(max_iters):
        c_node_list.append(c_node)
        key_list.append(self.keys[c_node])

        # if j == max_iters - 1:
        #   not_found[i] = True
        #   print(f"dead loop! {c_node_list=}", c_key, key_list, self.get_hash(c_key[None]), [self.get_hash(key[None]) for key in key_list])
          
        ### if the key at current node matches the given key, return the value at current node, exit.
        if torch.allclose(c_key, self.keys[c_node]):
          values[i] = self.values[c_node]
          # print(f"found {c_node_list=}", c_key, self.get_hash(c_key[None]), [self.get_hash(key[None]) for key in key_list])
          break

        ### if not, if the next node doesn't exist, set not_found to True, exit.
        if self.next_node[c_node].item() == -1:
          key_list.append(self.keys[self.next_node[c_node]])
          # print(f"not found {c_node_list=}", c_key, key_list, self.get_hash(c_key[None]), [self.get_hash(key[None]) for key in key_list])
          not_found[i] = True
          # print(f"not found wtf {depth}")
          break

        ### if the next node exists, set current node to next node
        # print(c_node, self.next_node[c_node])
        c_node = self.next_node[c_node]
        depth += 1
    
    return values, not_found

  def get(
      self, 
      keys : torch.Tensor,  # [n, *key_shape]
    ):
    keys = keys.clone()
    if self.keys is None:      
      return None, None

    assert keys.dtype == self.keys.dtype
    assert keys.device == self.keys.device
    assert keys.shape[1:] == self.keys.shape[1:]

    batch_size = keys.shape[0]
    values = torch.zeros(
      batch_size, 
      *self.values.shape[1:],
      device=self.values.device,
      dtype=self.values.dtype,
    ) - 12
    
    current_node = self.get_hash(keys) #[n]
    ### First, if the head node is empty, then set not_found to True
    not_found = self.head_node_is_empty[current_node]
    found = torch.zeros_like(not_found)

    # for i in range(20):
    while True:
      not_found_yet = ~(not_found ^ found)
      # print(f"{not_found=}")
      # print(f"{found=}")
      # print(not_found_yet, not_found_yet.any())
      # not_found_yet = ~(not_found | found)

      if not (not_found_yet).any():
        break
      ### if the key at current node matches the given key, return the value at current node, exit.
      does_key_match = self.all_close(self.keys[current_node], keys)
      just_found = does_key_match & not_found_yet

      values[just_found] = self.values[current_node[just_found]]
      found[just_found] = True
      # not_found_yet = ~(not_found | found)

      ### if not, if the next node doesn't exist, set not_found to True, exit.
      next_node_dont_exist = (self.next_node[current_node] == -1)
      not_found[not_found_yet & (~does_key_match) & next_node_dont_exist] = True

      ### if the next node exists, set current node to next node
      proceed_to_next_node = not_found_yet & (~does_key_match) & (~next_node_dont_exist)
      current_node[proceed_to_next_node] = self.next_node[current_node[proceed_to_next_node]]
      # not_found_yet = ~(not_found | found)
    
    return values, not_found

  def set_naive(
      self,
      keys,
      values
    ):
    keys = keys.clone()
    values = values.clone()
    start, stop = self.timer.create_context("set_naive")
    start("prep")
    if self._keys is None:
      self._keys = torch.empty(
        self.capacity,
        *keys.shape[1:],
        device=keys.device,
        dtype=keys.dtype
      )
      self._values = torch.empty(
        self.capacity,
        *values.shape[1:],
        device=values.device,
        dtype=values.dtype,
      )
    else:
      assert keys.dtype == self.keys.dtype
      assert keys.device == self.keys.device
      assert keys.shape[1:] == self.keys.shape[1:]
      assert values.dtype == self.values.dtype
      assert values.device == self.values.device
      assert values.shape[1:] == self.values.shape[1:]
    stop("prep")

    ### deduplicate keys
    key_is_unique = unique_first_occurrence(keys, dim=0)
    keys = keys[key_is_unique]
    values = values[key_is_unique]

    start("get_hash")
    ### get the index of they keys on the hash table
    current_node = self.get_hash(keys) #[n]
    stop("get_hash")

    start("ufo")
    ### check whether the hash code is unique first occurrence in current batch
    current_node_is_ufo = unique_first_occurrence(current_node, return_indices=False)
    stop("ufo")

    ### check whether a head node of a hash code is empty
    start("store_in_head_node")
    head_node_is_empty = self.head_node_is_empty[current_node] #[n], bool

    ### if both of the previous conditions are satisfied
    ### then store the key and value at head node
    store_in_head_node = head_node_is_empty & current_node_is_ufo
    stop("store_in_head_node")

    ### store key and value
    start("store key and value")
    self.keys[current_node[store_in_head_node]] = keys[store_in_head_node]
    self.values[current_node[store_in_head_node]] = values[store_in_head_node]

    ### set the head node as occupied, since we just stored data in it
    self.head_node_is_empty[current_node[store_in_head_node]] = False
    
    ### increment total element count
    self._n_elements += store_in_head_node.sum().item()
    stop("store key and value")

    ### select keys and values that aren't stored in the head node. exit if none left
    current_node = current_node[~store_in_head_node]
    if len(current_node) == 0:
      return
    start("select key values")
    values = values[~store_in_head_node]
    keys = keys[~store_in_head_node]
    stop("select key values")

    ### loop until exit condition is met.
    max_iter = 1000
    for i in range(max_iter):
      ### check whether the given key is found in current node.
      start("check is key found")
      is_key_found = self.all_close(self.keys[current_node], keys)
      stop("check is key found")

      ### if the key is found, then replace the corresponding value with new value
      start("return value if key is found")
      self.values[current_node[is_key_found]] = values[is_key_found]
      stop("return value if key is found")
      
      ### select remaining keys and values (keys that are not found in current node)
      start("select key values again")
      current_node = current_node[~is_key_found]
      if len(current_node) == 0:
        break
      keys = keys[~is_key_found]
      values = values[~is_key_found]
      stop("select key values again")

      start("next node exist")
      ### check whether the next nodes exist
      next_nodes_exist = self.next_node[current_node] != -1
      # print(f"{next_nodes_exist.sum()=}")

      ### if next node exists, then proceed to next node
      current_node[next_nodes_exist] = self.next_node[current_node[next_nodes_exist]]
      stop("next node exist")

      start("is current node ufo")
      ### else, check whether the current node is the first unique occurrence in batch
      current_node_is_ufo = unique_first_occurrence(current_node, return_indices=False)
      stop("is current node ufo")

      ### if current node is first unique occurrence, create a new node, append it to the end of the storage
      start("store in new node, expand if necessary")
      store_in_new_node = current_node_is_ufo & (~next_nodes_exist)
      new_nodes = torch.arange(store_in_new_node.sum(), device=self.keys.device) + self.n_nodes

      while self.n_nodes + len(new_nodes) > self.capacity:
        self.expand()
        
      ### increment number of elements stored in hash table 
      self._n_elements += len(new_nodes)

      ### increment number of nodes in hash table 
      self._n_nodes += len(new_nodes)

      ### store key and value in the new node
      self.keys[new_nodes] = keys[store_in_new_node]
      self.values[new_nodes] = values[store_in_new_node]
      
      ### link new node to next node (current_node.next_node = &new_node)
      self.next_node[current_node[store_in_new_node]] = new_nodes
      stop("store in new node, expand if necessary")

      ### select remaining nodes (node whose next node doesn't exist and current node isn't first unique occurrence, or next node exists)
      ### as well as corresponding key and value
      current_node = current_node[~store_in_new_node]
      if len(current_node) == 0:
        break
      start("select key values third time")
      keys = keys[~store_in_new_node]
      values = values[~store_in_new_node]
      stop("select key values third time")

  def set(
      self, 
      keys : torch.Tensor,  # [n, *key_shape]
      values : torch.Tensor # [n, *value_shape]
    ):
    start, stop = self.timer.create_context("set")
    start("prep")
    if self._keys is None:
      self._keys = torch.empty(
        self.capacity,
        *keys.shape[1:],
        device=keys.device,
        dtype=keys.dtype
      )
      self._values = torch.empty(
        self.capacity,
        *values.shape[1:],
        device=values.device,
        dtype=values.dtype,
      )
    else:
      assert keys.dtype == self.keys.dtype
      assert keys.device == self.keys.device
      assert keys.shape[1:] == self.keys.shape[1:]
      assert values.dtype == self.values.dtype
      assert values.device == self.values.device
      assert values.shape[1:] == self.values.shape[1:]
    stop("prep")

    start("get_hash")
    current_node = self.get_hash(keys) #[n]

    stop("get_hash")

    start("head_node_is_empty")
    head_node_is_empty = self.head_node_is_empty[current_node] #[n], bool
    current_node_is_ufo = unique_first_occurrence(current_node, return_indices=False)
    store_in_head_node = current_node_is_ufo & head_node_is_empty
    stop("head_node_is_empty")

    ### if empty:
    start("part 1")
    self.keys[current_node[store_in_head_node]] = keys[store_in_head_node]
    self.values[current_node[store_in_head_node]] = values[store_in_head_node]
    self.head_node_is_empty[current_node[store_in_head_node]] = False
    self._n_elements += store_in_head_node.sum().item()
    stop("part 1")

    ### else:
    start("part 2")
    current_node = current_node[~store_in_head_node]
    if len(current_node) == 0:
      return
    next_nodes = self.next_node[current_node]
    # prev_nodes = self.prev_node[current_node[~head_node_is_empty]]
    values = values[~store_in_head_node]
    keys = keys[~store_in_head_node]
    stop("part 2")

    max_iter = 100
    # while True:
    for i in range(max_iter):
      ### compare key with current node's key
      # self.values[current_node] - values[]
      start("part 3")
      
      is_key_found = self.all_close(self.keys[current_node], keys)
      stop("part 3")

      ### if the keys is found, then replace the corresponding value with new value, exit
      start("part 4")
      self.values[current_node[is_key_found]] = values[is_key_found]
      
      current_node = current_node[~is_key_found]
      stop("part 4")

      if len(current_node) == 0:
        break
      start("part 5")
      next_nodes = self.next_node[current_node]
      keys = keys[~is_key_found]
      values = values[~is_key_found]
      stop("part 5")

      ### if the key isn't found, then look for next node
      ### if the next node exists, set current node to next node
      start("part 6")
      next_nodes_exist = next_nodes != -1
      current_node[next_nodes_exist] = next_nodes[next_nodes_exist]
      stop("part 6")

      ### else if the current node is first unique occurrence, create new node, set next node to new node, exit
      start("part 7")
      current_node_is_ufo = unique_first_occurrence(current_node, return_indices=False)
      store_in_new_node = (~next_nodes_exist) & current_node_is_ufo
      new_nodes = torch.arange(store_in_new_node.sum(), device=self.keys.device) + len(self)

      self.next_node[current_node[store_in_new_node]] = new_nodes
      self.keys[new_nodes] = keys[store_in_new_node]
      self.values[new_nodes] = values[store_in_new_node]
      self._n_elements += len(new_nodes)
      
      current_node = current_node[~store_in_new_node]
      if len(current_node) == 0:
        break
      keys = keys[~store_in_new_node]
      values = values[~store_in_new_node]
      next_nodes = self.next_node[current_node]
      stop("part 7")
      # print(current_node)
      #      
  def test_uniformity(self, keys):
    hash_codes = self.get_hash(keys)
    logits = (torch.bincount(hash_codes, minlength=self.n_buckets) + 1).log()
    def entropy(logits):
      logp = torch.log_softmax(logits, dim=-1)
      p = torch.softmax(logits, dim=-1)
      return -(logp * p).sum(dim=-1)
    return entropy(logits).exp() / self.n_buckets

  def remove(self, keys):
    pass
