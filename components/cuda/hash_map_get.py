
from typing import Union
from pathlib import Path
import torch
import cupy as cp
import math
from UWOT.util import str2dtype, dtype2ctype, get_absolute_path
from UWOT.components.cuda_callable import CudaCallable

class HashMapGetCuda(CudaCallable):
  def __init__(
      self,
      tpb : int = 32,
      stages : int = 2,
      verbose = 0,
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
    self.tpb = tpb
    self.stages = stages

    kernel_name = "hash_map_get"
    # kernel_root = Path("F:/Python Projects/UWOT/components/cuda")
    # with open(kernel_root / f"{kernel_name}.cu", "r") as f:
    #   self.kernel = f.read()
    with open(get_absolute_path("components", "cuda", f"{kernel_name}.cu"), "r") as f:
      self.kernel = f.read()

    self.kernel = (self.kernel
      .replace("_TPB_", str(self.tpb))
      .replace("_STAGES_", str(self.stages))
    )

    self.fn = cp.RawKernel(
      self.kernel,
      f"hash_map_get",
      backend="nvcc",
      # options=[],
    )

    self.fn.max_dynamic_shared_size_bytes = 48 * 1024
    if verbose > 0:
      print(self.fn.attributes)

  def __call__(
      self, 
      query_hash_code : torch.Tensor,
      query_uuid : torch.Tensor,
      uuid : torch.Tensor,
      next_node : torch.Tensor,
      head_node : torch.Tensor
    ):
    assert query_hash_code.device == query_uuid.device == uuid.device == next_node.device == head_node.device == self.device
    assert query_hash_code.dtype == query_uuid.dtype == uuid.dtype == head_node.dtype == torch.long
    assert query_hash_code.shape == query_uuid.shape
    assert uuid.shape == next_node.shape
    assert query_hash_code.is_contiguous()
    assert query_uuid.is_contiguous()
    assert uuid.is_contiguous()
    assert next_node.is_contiguous()
    assert head_node.is_contiguous()

    n_query = query_uuid.shape[0]
    n = uuid.shape[0]

    indices = torch.zeros(n_query, device=self.device, dtype=torch.long)
    is_key_found = torch.zeros(n_query, device=self.device, dtype=torch.uint8)
    
    blocks_per_grid = ( math.ceil(n_query / (self.tpb * self.stages) ) ,)
    threads_per_block = (self.tpb, )

    self.fn(
      grid=blocks_per_grid,
      block=threads_per_block,
      args=[
        query_hash_code.data_ptr(),
        query_uuid.data_ptr(),
        uuid.data_ptr(),
        next_node.data_ptr(),
        head_node.data_ptr(),
        indices.data_ptr(),
        is_key_found.data_ptr(),
        n_query,
      ],
      stream = self.stream,
    )
    return indices, is_key_found.bool()

