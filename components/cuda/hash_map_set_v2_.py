
import math
from typing import Union
from pathlib import Path
import torch
import cupy as cp
from UWOT.util import str2dtype, dtype2ctype, get_absolute_path
from UWOT.components.cuda_callable import CudaCallable

class HashMapSetCuda_v2(CudaCallable):
  def __init__(
      self,
      key_type : Union[str, torch.dtype],
      key_per_block : int = 1,
      tpb_p1 : int = 32,
    ):
    """
      TODO:
        1. increase threads per block for better occupancy
        2. multiple new keys per tb
        3. prefetch.
        4. is there anything that can be coalesced?
          - hash codes
          - new keys
          - everthing else is random access
        5. how to distribute the workload per thread block?
          - each thread is responsible for one hash code, 32 at a time
          - each K threads are responsible for one hash code, 32 / K at a time
          - entire warp is responsible for one hash code, 1 at a time
          - 
    """
    super().__init__()
    self.key_type = str2dtype(key_type)
    self.key_type_str = dtype2ctype(self.key_type)
    self.tpb_p1 = tpb_p1
    self.key_per_block = key_per_block

    kernel_name = "hash_map_set_v2"
    # kernel_root = Path("F:/Python Projects/UWOT/components/cuda")
    # with open(kernel_root / f"{kernel_name}.cu", "r") as f:
    #   self.kernel = f.read()
    with open(get_absolute_path("components", "cuda", f"{kernel_name}.cu"), "r") as f:
      self.kernel = f.read()

    self.kernel = (self.kernel
      .replace("_KEYTYPE_", self.key_type_str)
      .replace("_TPB_", str(self.tpb_p1))
      .replace("_KPB_", str(self.key_per_block))
    )

    self.fn_p1 = cp.RawKernel(
      self.kernel,
      f"hash_map_set_p1",
      backend="nvcc",
      # options=[],
    )
    self.fn_p2 = cp.RawKernel(
      self.kernel,
      f"hash_map_set_p2",
      backend="nvcc",
      # options=[],
    )
    self.fn_p1.max_dynamic_shared_size_bytes = 48 * 1024

  def call_p1(
      self,
      new_keys : torch.Tensor,
      hash_codes : torch.Tensor,
      keys : torch.Tensor,
      next_node : torch.Tensor,
      head_node_is_empty : torch.Tensor,
    ):
    assert new_keys.device == keys.device == hash_codes.device == next_node.device == head_node_is_empty.device == self.device
    assert new_keys.dtype == keys.dtype == self.key_type
    assert new_keys.shape[0] == hash_codes.shape[0]
    assert keys.shape[0] == next_node.shape[0]
    assert keys.shape[1:] == new_keys.shape[1:]
    assert keys.is_contiguous()
    assert new_keys.is_contiguous()
    assert hash_codes.is_contiguous()
    assert next_node.is_contiguous()
    assert head_node_is_empty.is_contiguous()

    n = keys.shape
    n_new = new_keys.shape[0]
    n_buckets = head_node_is_empty.shape[0]
    key_size = new_keys.shape[1]
    keys = keys.view(n, -1)
    new_keys = new_keys.view(n_new, -1)
    key_size = keys.shape[1]

    last_visited_node = torch.empty(n_new, device=new_keys.device, dtype=torch.long)
    status = torch.zeros(n_new, device=new_keys.device, dtype=torch.uint8)
    self.fn_p1(
      grid=(math.ceil(n_new / self.key_per_block),),
      block=(self.tpb_p1,),
      args=[
        new_keys.data_ptr(),
        hash_codes.data_ptr(),
        keys.data_ptr(),
        next_node.data_ptr(),
        head_node_is_empty.data_ptr(),
        last_visited_node.data_ptr(),
        status.data_ptr(),
        key_size,
      ],
      stream = self.stream,
      shared_mem = key_size * keys.element_size(),
      
    )
    return last_visited_node, status

  def call_p2(
      self,
      new_node : torch.Tensor,
      last_visited_node : torch.Tensor,
      next_node : torch.Tensor,
    ):
    assert new_node.device == last_visited_node.device == next_node.device == self.device
    assert new_node.shape == last_visited_node.shape
    assert new_node.dtype == last_visited_node.dtype == next_node.dtype == torch.long
    assert new_node.is_contiguous()
    assert last_visited_node.is_contiguous()
    assert next_node.is_contiguous()
    n_new = new_node.shape[0]

    unique_new_node, inverse = new_node.unique(return_inverse=True)
    mutex = torch.zeros_like(unique_new_node, dtype=torch.int32, device=new_node.device)
    self.fn_p2(
      grid=(n_new,),
      block=(32,),
      args=[
        new_node.data_ptr(),
        last_visited_node.data_ptr(),
        next_node.data_ptr(),
        inverse.data_ptr(),
        mutex.data_ptr(),
      ],
      stream = self.stream
    )

