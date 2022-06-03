import torch
import torch.nn.functional as F
from typing import Union, Optional
from torch import Tensor

from .components.cuda.topkspspcsim import TopkSPSPCSIM


def support_dense(func):
  func_name = func.__name__
  if hasattr(torch, func_name):
    torch_func = getattr(torch, func_name)

  elif hasattr(F, func_name):
    torch_func = getattr(F, func_name)

  else:
    return func

  def new_func(*args, **kwargs):
    all_tensors_in_args = [i for i in args if isinstance(i, Tensor)]
    all_tensors_in_kwargs = [i for i in kwargs.values() if isinstance(i, Tensor)]
    all_tensors = all_tensors_in_args + all_tensors_in_kwargs
    all_dense = all([not tensor.layout in {torch.sparse_coo, torch.sparse_csr} for tensor in all_tensors])
    if all_dense:
      return torch_func(*args, **kwargs)
    return func(*args, **kwargs)

  return new_func

@support_dense
def sum(
    input : Tensor,
    dim : Optional[Union[int, tuple[int], list[int]]] = None, 
    keepdim : bool = False,
  ) -> Tensor :
  # if not tensor.is_sparse:
  #   return torch.sum(tensor, dim=dim, keepdim=keepdim)

  y = torch.sparse.sum(input, dim=dim)
  
  if keepdim:
    if isinstance(dim, int):
      y = y.unsqueeze(dim=dim)

    elif isinstance(dim, (tuple, list)):
      dim = sorted(dim, reverse=False)
      for d in dim:
        y = y.unsqueeze(dim=d)

    elif isinstance(dim, None):
      for d in range(input.ndim):
        y.unsqueeze(dim=0)

  return y

@support_dense
def norm(
    input : Tensor, 
    dim : Optional[Union[int, tuple[int], list[int]]] = None,
    p : float = 2, 
    keepdim : bool = False,
  ) -> Tensor:
  # if not tensor.is_sparse:
  #   return torch.norm(tensor, dim=dim, p=p, keepdim=keepdim)

  if p == 0:
    new_values = input.values().bool().to(input.dtype)
  elif p == 1:
    new_values = input.values().abs()
  elif p == 2:
    new_values = input.values().square()
  else:
    new_values = input.values().abs().pow(p)
  if input.layout == torch.sparse_coo:
    input = torch.sparse_coo_tensor(
      indices=input.indices(),
      values=new_values,
      size=input.shape,
      dtype=input.dtype,
      device=input.device,
    )
  elif input.layout == torch.sparse_csr:
    torch.sparse_csr_tensor(
      crow_indices=input.crow_indices(),
      col_indices=input.col_indices(),
      values=new_values,
      size=input.shape,
      dtype=input.dtype,
      device=input.device
    )
  y = input.sum(dim=dim, keepdim=keepdim)
  if p == 2:
    y = y.sqrt()
  else:
    y = y.pow(1 / p)

  return y

@support_dense
def softmax(
    input : Tensor, 
    dim : int = -1,
  ) -> Tensor:
  dim = dim % input.ndim
  # if not tensor.is_sparse:
  #   return torch.softmax(tensor, dim=dim)

  return torch.sparse.softmax(input, dim=dim)

@support_dense
def log_softmax(
    input : Tensor, 
    dim : int = -1,
  ) -> Tensor:
  dim = dim % input.ndim
  # if not tensor.is_sparse:
  #   return torch.log_softmax(tensor, dim=dim)

  return torch.sparse.log_softmax(input, dim=dim)

@support_dense
def normalize(
    input : Tensor,
    dim : int = 0,
    p : float = 2,
    eps : float = 1e-12,
  ) -> Tensor:
  # if not input.is_sparse:
  #   return F.normalize(input, dim=dim, p=p, eps=eps)
  dim = dim % input.ndim
  if input.layout == torch.sparse_csr:
    input = input.to_sparse_coo()
    
  if not input.is_coalesced():
    input = input.coalesce()
  
  if p == 0:
    new_values = input.values().bool().to(input.dtype)
  elif p == 1:
    new_values = input.values().abs()
  elif p == 2:
    new_values = input.values().square()
  else:
    new_values = input.values().abs().pow(p)
  new_values.clamp_min_(eps)
  new_values = new_values.log()

  input = torch.sparse_coo_tensor(
    indices=input.indices(),
    values=new_values,
    size=input.shape,
    dtype=input.dtype,
    device=input.device,
  )

  y = softmax(input, dim=dim)
  if p == 2:
    return y.sqrt()
  else:
    return y.pow(1 / p)

def normalize_(
    input : Tensor,
    dim : int = -1,
    p : float = 2,
    eps : float = 1e-12,
  ):
  dim = dim % input.ndim
  if input.is_sparse or input.is_sparse_csr:
    y = normalize(input=input, dim=dim, p=p, eps=eps)
    input.values()[:] = y.values()
  else:
    input_norm = input.norm(p=p, dim=dim, keepdim=True) + eps
    input.div_(input_norm)
  return input

def freq2prob(freqs : Tensor, dim=-1) -> Tensor:
  return normalize(freqs, dim=dim, p=1)

def cross_entropy(probs : Tensor, labels : Tensor, reduction="mean"):
  """
    probs : Tensor, shape = [n, v]
    labels : Tensor, shape = [n]
  """
  ce = -probs.gather(index=labels[:, None], dim=-1).log().flatten()
  # print(f"{ce=}")
  if reduction == "sum":
    ce = ce.sum()
  elif reduction == "mean":
    ce = ce.mean()
  elif reduction == "none":
    pass

  return ce

topkspsp_inner = TopkSPSPCSIM(sim_type="inner")
topkspsp_nl1 = TopkSPSPCSIM(sim_type="nl1")
topkspsp_nl2 = TopkSPSPCSIM(sim_type="nl2")

def sparse_topk_inner(a : Tensor, b : Tensor, dim=1, k=1):
  assert b.ndim == a.ndim == 2
  dim = dim % a.ndim
  assert dim == 1
  assert a.layout in {torch.sparse_coo, torch.sparse_csr}
  assert b.layout in {torch.sparse_coo, torch.sparse_csr}
  return topkspsp_inner(a, b, k=k)

def sparse_topk_nl1(a : Tensor, b : Tensor, dim=1, k=1):
  assert b.ndim == a.ndim == 2
  dim = dim % a.ndim
  assert dim == 1
  assert a.layout in {torch.sparse_coo, torch.sparse_csr}
  assert b.layout in {torch.sparse_coo, torch.sparse_csr}
  return topkspsp_nl1(a, b, k=k)

def sparse_topk_nl2(a : Tensor, b : Tensor, dim=1, k=1):
  assert b.ndim == a.ndim == 2
  dim = dim % a.ndim
  assert dim == 1
  assert a.layout in {torch.sparse_coo, torch.sparse_csr}
  assert b.layout in {torch.sparse_coo, torch.sparse_csr}
  return topkspsp_nl2(a, b, k=k)

def sparse_topk_cos(a : Tensor, b : Tensor, dim=1, k=1):
  assert b.ndim == a.ndim == 2
  dim = dim % a.ndim
  assert dim == 1
  assert a.layout in {torch.sparse_coo, torch.sparse_csr}
  assert b.layout in {torch.sparse_coo, torch.sparse_csr}
  a = normalize(a, dim=-1)
  b = normalize(b, dim=-1)
  return topkspsp_inner(a, b, k=k)
