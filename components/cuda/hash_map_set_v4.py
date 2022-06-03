
from typing import Union
from pathlib import Path
import torch
import cupy as cp
import math
from UWOT.components.cuda_callable import CudaCallable
from UWOT.util import str2dtype, dtype2ctype, get_absolute_path

class HashMapSetCuda(CudaCallable):
  def __init__(
      self,
      tpb : int = 32,
      stages : int = 2,
      verbose = 0,
    ):

    super().__init__()
    self.tpb = tpb
    self.stages = stages

    kernel_name = "hash_map_set_v4"
    # kernel_root = Path("F:/Python Projects/UWOT/components/cuda")
    with open(get_absolute_path("components", "cuda", f"{kernel_name}.cu"), "r") as f:
      self.kernel = f.read()

    self.kernel = (self.kernel
      .replace("_TPB_", str(self.tpb))
      .replace("_STAGES_", str(self.stages))
    )

    self.fn_p1 = cp.RawKernel(
      self.kernel,
      f"hash_map_set_p1",
      backend="nvcc",
      # options=[],
    )
    if verbose > 0:
      print(self.fn_p1.attributes)
    self.fn_p2 = cp.RawKernel(
      self.kernel,
      f"hash_map_set_p2",
      backend="nvcc",
      # options=[],
    )
    self.fn_p1.max_dynamic_shared_size_bytes = 48 * 1024
    if verbose > 0:
      print(self.fn_p2.attributes)

  def call_p1(
      self,
      new_hash_code : torch.Tensor,
      new_uuid : torch.Tensor,
      uuid : torch.Tensor,
      next_node : torch.Tensor,
      head_node : torch.Tensor,
    ):
    assert new_hash_code.device == next_node.device == uuid.device == new_uuid.device == head_node.device == self.device
    assert new_uuid.shape == new_hash_code.shape
    assert new_hash_code.is_contiguous()
    assert new_uuid.is_contiguous()
    assert uuid.is_contiguous()
    assert next_node.is_contiguous()
    assert head_node.is_contiguous()

    n = uuid.shape[0]
    n_new = new_uuid.shape[0]
    
    last_visited_node = torch.empty(n_new, device=self.device, dtype=torch.long)
    status = torch.zeros(n_new, device=self.device, dtype=torch.uint8)

    blocks_per_grid = ( math.ceil(n_new / (self.tpb * self.stages) ) ,)
    threads_per_block = (self.tpb, )
    self.fn_p1(
      grid=blocks_per_grid,
      block=threads_per_block,
      args=[
        new_hash_code.data_ptr(),
        new_uuid.data_ptr(),
        uuid.data_ptr(),
        next_node.data_ptr(),
        head_node.data_ptr(),
        last_visited_node.data_ptr(),
        status.data_ptr(),
        n_new,
      ],
      stream = self.stream,
      # shared_mem = key_size * keys.element_size(),
      
    )
    return last_visited_node, status

  def call_p2(
      self,
      new_node : torch.Tensor,
      last_visited_node : torch.Tensor,
      hash_code : torch.Tensor,
      next_node : torch.Tensor,
      head_node : torch.Tensor,
    ):
    assert new_node.device == last_visited_node.device == next_node.device == hash_code.device == head_node.device == self.device
    assert new_node.shape == last_visited_node.shape == hash_code.shape
    assert new_node.dtype == last_visited_node.dtype == next_node.dtype == hash_code.dtype == head_node.dtype == torch.long
    assert new_node.is_contiguous()
    assert last_visited_node.is_contiguous()
    assert next_node.is_contiguous()
    assert hash_code.is_contiguous()
    assert head_node.is_contiguous()
    n_new = new_node.shape[0]

    # unique, inverse = last_visited_node.unique(return_inverse=True)
    unique, inverse = hash_code.unique(return_inverse=True)
    mutex = torch.zeros_like(unique, dtype=torch.int32, device=new_node.device)
    
    blocks_per_grid = ( math.ceil(n_new / (self.tpb) ) ,)
    threads_per_block = (self.tpb, )

    self.fn_p2(
      grid=blocks_per_grid,
      block=threads_per_block,
      args=[
        new_node.data_ptr(),
        last_visited_node.data_ptr(),
        hash_code.data_ptr(),
        next_node.data_ptr(),
        head_node.data_ptr(),
        inverse.data_ptr(),
        mutex.data_ptr(),
        n_new,
      ],
      stream = self.stream
    )

