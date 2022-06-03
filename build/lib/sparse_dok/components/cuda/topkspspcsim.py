from collections import namedtuple
import cupy as cp
import torch
import numpy as np
import math
from torch import Tensor
from typing import Union
import sympy
from torchtimer import ProfilingTimer

from ..cuda_callable import CudaCallable
from ...util import get_absolute_path

ProblemSize = namedtuple("ProblemSize", ["m", "n", "k"])

class TopkSPSPCSIMCuda(CudaCallable):
  """
    Topk sparse sparse cross similarity kernel
  """
  def __init__(self, tpb=256, maxnnzpr=8, minnnzpr=0, tile_m=1, stack_cap=2, bin_search_ver=3, threads_per_group=32, sim_type="inner"):
    super().__init__()
    assert 32 <= tpb <= 1024
    self.tpb = tpb
    self.maxnnzpr = maxnnzpr
    self.minnnzpr = minnnzpr
    self.tile_m = tile_m
    self.stack_cap = stack_cap
    self.num_buckets = self.tile_m * self.maxnnzpr
    self.bin_search_ver = bin_search_ver
    self.threads_per_group = threads_per_group

    self.p = 1.0
    if sim_type in {"dot", "inner"}:
      self.sim_type = "SIM_INNER"

    elif sim_type in {"nl2", "negative_euclidean"}:
      self.sim_type = "SIM_NL2"
    
    elif sim_type in {"nl1", "negative_manhattan"}:
      self.sim_type = "SIM_NL1"
    
    elif sim_type.startswith("nl"):
      self.sim_type = "SIM_NLP"
      self.p = float(sim_type.replace("nl", ""))
    
    else:
      raise NotImplementedError(f"unsupported sim_fn {sim_type}")

    self.smem_size = self.tpb * 12 + self.tile_m * 8 + self.num_buckets * 12
    assert self.smem_size <= cp.cuda.runtime.getDeviceProperties(self.device.index)["sharedMemPerBlock"]

    cu_files = [
      "head.cu",
      "bitonic.cu",
      "smem_tensor.cu",
      "stack.cu",
      "binary_search.cu",
      # "smem_hashmap.cu",
      "reduce.cu",
      "topkspspcsim.cu",
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
      .replace("_SIMTYPE_", str(self.sim_type))
      .replace("_P_", str(self.p))
      .replace("int64_t", "ll_t")
    )

    self.fn = cp.RawKernel(
      self.kernel,
      "topkspspcsim",
      backend="nvcc",
      options=(
        '-std=c++17',
        "--device-c",
      )
    )
    self.fn.max_dynamic_shared_size_bytes = self.smem_size
    # print(self.fn.attributes)
    
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
    a_row_start = a_crow_inds[:-1]
    a_row_nnz = a_crow_inds[1:] - a_row_start
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
        a_row_start.data_ptr(),
        a_row_nnz.data_ptr(),
        a_col_inds.data_ptr(),
        a_vals.data_ptr(),

        b_row_start_sorted.data_ptr(),
        b_row_nnz_sorted.data_ptr(),
        b_col_inds.data_ptr(),
        b_vals.data_ptr(),
        
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
      problem_size : ProblemSize,
      a_row_start : Tensor,
      a_row_nnz : Tensor,
      a_col_inds : Tensor,
      a_vals : Tensor,

      b_row_start : Tensor,  #[n]
      b_row_nnz : Tensor, #[n]
      b_col_inds : Tensor, #[n]
      b_vals : Tensor, #[n]

      k : int = 256,
    ):
    m_ = a_row_start.shape[0]

    sub_topk_inds = torch.empty(m_, self.tpb, device=a_row_start.device, dtype=torch.long).fill_(-99)
    sub_topk_vals = torch.empty(m_, self.tpb, device=a_row_start.device, dtype=torch.float).fill_(float("-inf"))
    
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

        b_row_start.data_ptr(),
        b_row_nnz.data_ptr(),
        b_col_inds.data_ptr(),
        b_vals.data_ptr(),
        
        sub_topk_inds.data_ptr(),
        sub_topk_vals.data_ptr(),
        m_, problem_size.n, problem_size.k
      ],
      stream=self.stream,
      shared_mem=self.smem_size,
    )
    sub_topk_vals = sub_topk_vals[:, :k]
    sub_topk_inds = sub_topk_inds[:, :k]

    return sub_topk_vals, sub_topk_inds 


class TopkSPSPCSIM:
  def __init__(self, sim_type):
    # self.nnzpr_ranges = [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    # self.tile_ms =      [4, 4, 4, 4, 4,  4,  4,  4,   3,   1,   1]
    # self.nnzpr_ranges = [0, 4, 16, 64, 256, 512, 1024, 2048]
    # self.tile_ms =      [   4, 4,  4,  4,   3,   1,    1]
    self.nnzpr_ranges = [0, 32, 256, 512, 1024, 2048]
    self.tile_ms =      [   4,  4,   3,   1,    1]
    self.tpbs = [128, 256, 512, 1024]
    self.timer = ProfilingTimer(False, name="TopkSPSPCSIM")
    for i in range(len(self.nnzpr_ranges) - 1):

      minnnzpr = self.nnzpr_ranges[i]
      maxnnzpr = self.nnzpr_ranges[i + 1]
      tile_m = self.tile_ms[i]
      for tpb in self.tpbs:
        fn = TopkSPSPCSIMCuda(
          tpb=tpb, 
          minnnzpr=minnnzpr, 
          maxnnzpr=maxnnzpr, 
          tile_m=tile_m, 
          threads_per_group=4, 
          sim_type=sim_type
        )
        setattr(self, f"fn_{tpb}_{minnnzpr}_{maxnnzpr}", fn)

  def __call__(self, a : Tensor, b : Tensor, k=128, n_candidates=None):
    assert isinstance(a, Tensor)
    assert isinstance(b, Tensor)
    assert a.ndim == b.ndim == 2
    assert a.shape[1] == b.shape[1]
    assert a.layout in { torch.sparse_csr, torch.sparse_coo}
    assert b.layout in { torch.sparse_csr, torch.sparse_coo}
    assert a.device == b.device
    assert a.dtype == b.dtype == torch.float
    if n_candidates is not None:
      k = n_candidates
    start, stop = self.timer.create_context("__call__")

    start("to sparse csr")
    assert 0 < k <= 1024
    if a.layout == torch.sparse_coo:
      a = a.to_sparse_csr()
    
    if b.layout == torch.sparse_coo:
      b = b.to_sparse_csr()
    stop("to sparse csr")
    # problem_size = (a.shape[0], *b.shape)
    start("init topk inds and values")
    problem_size = ProblemSize(m=a.shape[0], n=b.shape[0], k=b.shape[1])
    topk_inds = torch.empty(problem_size.m, k, device=a.device, dtype=torch.long).fill_(-99)
    topk_vals = torch.empty(problem_size.m, k, device=a.device, dtype=torch.float).fill_(float("-inf"))
    stop("init topk inds and values")
    
    start("get a crow col val")
    a_crow_inds = a.crow_indices()
    a_row_start = a_crow_inds[:-1]
    a_row_nnz = a_crow_inds[1:] - a_row_start
    a_col_inds = a.col_indices()
    a_vals = a.values()
    stop("get a crow col val")

    start("get b crow col val")
    b_crow_inds = b.crow_indices()
    b_col_inds = b.col_indices()
    b_vals = b.values()
    b_row_start = b_crow_inds[:-1]
    b_row_nnz = b_crow_inds[1:] - b_row_start
    sorted_b_row_nnz, sorted_indices = torch.sort(b_row_nnz)
    sorted_b_row_start = b_row_start[sorted_indices]
    stop("get b crow col val")
    
    if k <= 128:
      tpb = 128
    elif k <= 256:
      tpb = 256
    elif k <= 512:
      tpb = 512
    elif k <= 1024:
      tpb = 1024
    max_a_row_nnz = a_row_nnz.max().item()
    for i in range(len(self.nnzpr_ranges)-1):
      start("misc and mask")
      minnnzpr = self.nnzpr_ranges[i]
      maxnnzpr = self.nnzpr_ranges[i+1]
      if max_a_row_nnz <= minnnzpr:
        break
      fn = getattr(self, f"fn_{tpb}_{minnnzpr}_{maxnnzpr}")
      if i == len(self.nnzpr_ranges)-2:
        mask = (minnnzpr < a_row_nnz)
      else:
        mask = (minnnzpr <= a_row_nnz) & (a_row_nnz < maxnnzpr)
      selected_a_row_nnz = a_row_nnz[mask]
      stop("misc and mask")
      if len(selected_a_row_nnz) == 0:
        continue
      start("kernel")
      selected_a_row_start = a_row_start[mask]
      sub_topk_vals, sub_topk_inds = fn.call(
        problem_size,
        selected_a_row_start,
        selected_a_row_nnz,
        a_col_inds,
        a_vals,
        sorted_b_row_start,
        sorted_b_row_nnz,
        b_col_inds,
        b_vals,
        k
      )
      stop("kernel")
      start("sub_topk_vals to topk_vals")
      topk_vals[mask] = sub_topk_vals
      topk_inds[mask] = sub_topk_inds
      stop("sub_topk_vals to topk_vals")
    start("final map")
    topk_inds = sorted_indices[topk_inds]
    stop("final map")
    return topk_vals, topk_inds

  def __del__(self):
    self.timer.summarize()