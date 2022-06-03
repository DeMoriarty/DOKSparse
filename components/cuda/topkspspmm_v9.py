import cupy as cp
import torch
import numpy as np
import math
from torch import Tensor
from typing import Union
import sympy

from UWOT.components.cuda_callable import CudaCallable
from UWOT.util import get_absolute_path

class TopkSPSPMMCuda(CudaCallable):
  def __init__(self, tpb=256, maxnnzpr=8, minnnzpr=0, tile_m=1, stack_cap=2, bin_search_ver=3, threads_per_group=32):
    super().__init__()
    assert 32 <= tpb <= 1024
    self.tpb = tpb
    self.maxnnzpr = maxnnzpr
    self.minnnzpr = minnnzpr
    self.tile_m = tile_m
    self.stack_cap = stack_cap
    self.bin_search_ver = bin_search_ver
    self.threads_per_group = threads_per_group
    self.num_buckets = int(self.tile_m * self.maxnnzpr * 4)

    self.primes1 = torch.tensor(sympy.nextprime(np.random.randint(self.num_buckets, self.num_buckets*2)), device=self.device, dtype=torch.long)
    self.primes2 = torch.tensor(sympy.nextprime(np.random.randint(self.num_buckets, self.num_buckets*2)), device=self.device, dtype=torch.long)
    self.alpha1 = torch.tensor(np.random.randint(self.primes1.cpu()) - 1, device=self.device, dtype=torch.long)
    self.alpha2 = torch.tensor(np.random.randint(self.primes2.cpu()) - 1, device=self.device, dtype=torch.long)
    self.beta1 = torch.tensor(np.random.randint(self.primes1.cpu()) - 1, device=self.device, dtype=torch.long)
    self.beta2 = torch.tensor(np.random.randint(self.primes2.cpu()) - 1, device=self.device, dtype=torch.long)

    self.smem_size = self.tpb * 12 + self.tile_m * 8 + self.num_buckets * 12
    assert self.smem_size <= cp.cuda.runtime.getDeviceProperties(self.device.index)["sharedMemPerBlock"]

    cu_files = [
      "head.cu",
      "bitonic.cu",
      "smem_tensor.cu",
      "stack.cu",
      "binary_search.cu",
      "smem_hashmap_simple.cu",
      "reduce.cu",
      "topkspspmm_v9.cu",
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
      .replace("_TILEM_", str(self.tile_m))
      .replace("_STACKCAP_", str(self.stack_cap))
      .replace("_BINSEARCHVER_", str(self.bin_search_ver))
      .replace("_TPG_", str(self.threads_per_group))
      .replace("_NUMBUCKETS_", str(self.num_buckets))
      .replace("_LOG2NUMBUCKETS_", str(int(math.log2(self.num_buckets))))
      .replace("int64_t", "ll_t")
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
    self.fn.max_dynamic_shared_size_bytes = self.smem_size
    print(self.fn.attributes)
    
  def __call__(
      self, 
      a : Tensor,  #[m, k]
      b : Tensor, #[n]
      n_candidates : int = 256,
    ):
    assert isinstance(a, Tensor)
    assert a.layout in {torch.sparse_csr}
    assert 1 <= n_candidates <= self.tpb
    m, k = a.shape
    n, _ = b.shape
      
    a_crow_inds = a.crow_indices()
    a_col_inds = a.col_indices()
    a_vals = a.values()

    b_crow_inds = b.crow_indices()
    b_col_inds = b.col_indices()
    b_vals = b.values()
    b_row_start = b_crow_inds[:-1]
    b_row_nnz = b_crow_inds[1:] - b_row_start

    b_row_nnz_sorted, sorted_indices = torch.sort(b_row_nnz)
    b_row_start_sorted = b_row_start[sorted_indices]


    topk_inds = torch.empty(m, self.tpb, device=a.device, dtype=torch.long).fill_(-99)
    topk_vals = torch.empty(m, self.tpb, device=a.device, dtype=torch.float).fill_(float("-inf"))
    
    blocks_per_grid = (math.ceil(m / self.tile_m), )
    threads_per_block = (self.tpb, )
    self.fn(
      grid=blocks_per_grid,
      block=threads_per_block,
      args=[
        a_crow_inds.data_ptr(),
        a_col_inds.data_ptr(),
        a_vals.data_ptr(),

        b_row_start_sorted.data_ptr(),
        b_row_nnz_sorted.data_ptr(),
        b_col_inds.data_ptr(),
        b_vals.data_ptr(),

        self.alpha1.data_ptr(),
        self.alpha2.data_ptr(),
        self.beta1.data_ptr(),
        self.beta2.data_ptr(),
        self.primes1.data_ptr(),
        self.primes2.data_ptr(),
        
        topk_inds.data_ptr(),
        topk_vals.data_ptr(),
        m, n, k
      ],
      stream=self.stream,
      shared_mem=self.smem_size,
    )
    topk_vals = topk_vals[:, :n_candidates]
    topk_inds = topk_inds[:, :n_candidates]
    topk_inds = sorted_indices[topk_inds]

    return topk_vals, topk_inds 

  def call(
      self, 
      a : Tensor,  #[m, k]
      b_row_start_sorted : Tensor,  #[n]
      b_row_nnz_sorted : Tensor, #[n]
      b_col_inds : Tensor, #[n]
      b_vals : Tensor, #[n]
      topk_vals : Tensor, #[m, n_candidates]
      topk_inds : Tensor, #[m, n_candidates]
      n_candidates : int = 256,
    ):
    assert isinstance(a, Tensor)
    assert a.layout in {torch.sparse_csr}
    assert 1 <= n_candidates <= self.tpb
    m, k = a.shape
    n = b_row_start_sorted.shape[0]
      
    a_crow_inds = a.crow_indices()
    a_row_start = a_crow_inds[:-1]
    a_row_nnz = a_crow_inds[1:] - a_row_start
    mask = (self.minnnzpr < a_row_nnz) & (a_row_nnz <= self.maxnnzpr)
    selected_a_row_indices = torch.nonzero(mask, as_tuple=True)[0]
    a_row_start = a_row_start[selected_a_row_indices]
    a_row_nnz = a_row_nnz[selected_a_row_indices]
    m_ = a_row_start.shape[0]
    if m_ == 0:
      return topk_vals, topk_inds
    a_col_inds = a.col_indices()
    a_vals = a.values()


    sub_topk_inds = torch.empty(m_, self.tpb, device=a.device, dtype=torch.long).fill_(-99)
    sub_topk_vals = torch.empty(m_, self.tpb, device=a.device, dtype=torch.float).fill_(float("-inf"))
    
    blocks_per_grid = (math.ceil(m_ / self.tile_m), )
    threads_per_block = (self.tpb, )
    self.fn(
      grid=blocks_per_grid,
      block=threads_per_block,
      args=[
        a_row_start.data_ptr(),
        a_row_nnz.data_ptr(),
        a_col_inds.data_ptr(),
        a_vals.data_ptr(),

        b_row_start_sorted.data_ptr(),
        b_row_nnz_sorted.data_ptr(),
        b_col_inds.data_ptr(),
        b_vals.data_ptr(),

        self.alpha1.data_ptr(),
        self.alpha2.data_ptr(),
        self.beta1.data_ptr(),
        self.beta2.data_ptr(),
        self.primes1.data_ptr(),
        self.primes2.data_ptr(),
        
        sub_topk_inds.data_ptr(),
        sub_topk_vals.data_ptr(),
        m_, n, k
      ],
      stream=self.stream,
      shared_mem=self.smem_size,
    )
    sub_topk_vals = sub_topk_vals[:, :n_candidates]
    sub_topk_inds = sub_topk_inds[:, :n_candidates]
    # 
    topk_vals[selected_a_row_indices] = sub_topk_vals
    topk_inds[selected_a_row_indices] = sub_topk_inds
    return topk_vals, topk_inds 


def topkspspmm(a : Tensor, b : Tensor, n_candidates=256):
  assert isinstance(a, Tensor)
  assert isinstance(b, Tensor)
  assert a.ndim == b.ndim == 2
  assert a.shape[1] == b.shape[1]
  assert a.layout in { torch.sparse_csr, torch.sparse_coo}
  assert b.layout in { torch.sparse_csr, torch.sparse_coo}
  assert a.device == b.device
  assert a.dtype == b.dtype == torch.float

  if a.layout == torch.sparse_coo:
    a = a.to_sparse_csr()
  
  if b.layout == torch.sparse_coo:
    b = b.to_sparse_csr()

  m, k = a.shape
  n, _ = b.shape
  topk_inds = torch.empty(m, n_candidates, device=a.device, dtype=torch.long).fill_(-99)
  topk_vals = torch.empty(m, n_candidates, device=a.device, dtype=torch.float).fill_(float("-inf"))

  b_crow_inds = b.crow_indices()
  b_col_inds = b.col_indices()
  b_vals = b.values()
  b_row_start = b_crow_inds[:-1]
  b_row_nnz = b_crow_inds[1:] - b_row_start

  b_row_nnz_sorted, sorted_indices = torch.sort(b_row_nnz)
  b_row_start_sorted = b_row_start[sorted_indices]

  # do stuff
  topk_inds = sorted_indices[topk_inds]