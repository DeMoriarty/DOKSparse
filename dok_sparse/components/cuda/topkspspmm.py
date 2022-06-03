import cupy as cp
import torch
import numpy as np
import math
from torch import Tensor
from typing import Union

from UWOT.components.cuda_callable import CudaCallable
from UWOT.components import SparseDOKTensor
from UWOT.util import get_absolute_path

class TopkSPSPMMCuda(CudaCallable):
  def __init__(self, tpb=256, maxnnzpr=8, a_rows_per_block=1):
    super().__init__()
    assert 32 <= tpb <= 1024
    self.tpb = tpb
    self.maxnnzpr = maxnnzpr
    self.a_rows_per_block = a_rows_per_block

    cu_files = [
      "head.cu",
      "bitonic.cu",
      "smem_tensor.cu",
      "hashmap.cu",
      "topkspspmm.cu",
    ]
      
    kernel = []
    for file in cu_files:
      with open(get_absolute_path("components", "cuda", file), "r") as f:
        kernel.append(f.read())

    self.kernel = "\n".join(kernel)

    with open(get_absolute_path("components", "cuda", "preview.cu"), "w") as f:
      f.write(self.kernel)

    self.kernel = (self.kernel
      .replace("_TPB_", str(self.tpb))
      .replace("_MAXNNZPR_", str(self.maxnnzpr))
      .replace("_ARPB_", str(self.a_rows_per_block))
    )

    self.fn = cp.RawKernel(
      self.kernel,
      "topkspspmm",
      backend="nvcc",
      options=(
        '-std=c++17',
        "--device-c",
      )
    )
    self.smem_size = self.a_rows_per_block * self.maxnnzpr * 12 + self.tpb * 12
    self.fn.max_dynamic_shared_size_bytes = self.smem_size
    
  def __call__(
      self, 
      a : Union[Tensor, SparseDOKTensor],  #[m, k]
      b : SparseDOKTensor,  #[n, k]
      n_candidates : int = 256,
    ):
    assert isinstance(a, (Tensor, SparseDOKTensor))
    assert isinstance(b, SparseDOKTensor), "second operand must be SparseDOKTensor"
    assert a.is_sparse, "first operand is not sparse"
    # assert isinstance(out, (Tensor, SparseDOKTensor))
    assert a.ndim == b.ndim == 2
    assert a.dtype == b.dtype == torch.float
    assert a.shape[-1] == b.shape[-1]
    assert 1 <= n_candidates <= self.tpb
    m, k = a.shape
    n, _ = b.shape

    if isinstance(a, SparseDOKTensor):
      a = a.to_sparse_coo()
    a = a.to_sparse_csr()
      
    a_crow_inds = a.crow_indices()
    a_col_inds = a.col_indices()
    a_vals = a.values().clone()
    alpha1 = b.storage().alpha1
    alpha2 = b.storage().alpha2
    beta1 = b.storage().beta1
    beta2 = b.storage().beta2
    primes1 = b.storage().primes1
    primes2 = b.storage().primes2
    head_node = b.storage().head_node
    next_node = b.storage().next_node
    uuid = b.storage().uuid
    b_vals = b.values().clone()
    n_buckets = b.storage().n_buckets

    topk_inds = torch.empty(m, self.tpb, device=a.device, dtype=torch.long)
    topk_vals = torch.empty(m, self.tpb, device=a.device, dtype=torch.float).fill_(float("-inf"))
    
    blocks_per_grid = (math.ceil(m / self.a_rows_per_block), )
    threads_per_block = (self.tpb, )
    self.fn(
      grid=blocks_per_grid,
      block=threads_per_block,
      args=[
        a_crow_inds.data_ptr(),
        a_col_inds.data_ptr(),
        a_vals.data_ptr(),
        alpha1.data_ptr(),
        alpha2.data_ptr(),
        beta1.data_ptr(),
        beta2.data_ptr(),
        primes1.data_ptr(),
        primes2.data_ptr(),
        head_node.data_ptr(),
        next_node.data_ptr(),
        uuid.data_ptr(),
        b_vals.data_ptr(),
        topk_inds.data_ptr(),
        topk_vals.data_ptr(),
        m, n, k, n_buckets
      ],
      stream=self.stream,
      shared_mem=self.smem_size,
    )
    topk_vals = topk_vals[:, :n_candidates]
    topk_inds = topk_inds[:, :n_candidates]
    return topk_vals, topk_inds 


    