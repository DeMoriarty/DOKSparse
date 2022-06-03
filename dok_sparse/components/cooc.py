import torch
import torch.nn as nn

class Coocurrence(nn.Module):
  def __init__(
      self,
      n,
    ):
    super().__init__()
    weight = torch.zeros(n, n, dtype=torch.float)
    self.register_buffer("weight", weight)
    

  def left(self, idx=None, raw=True, center=True):
    """
      given right token, return left frequency/repr
    """
    mat = self.weight.T.clone()
    if not raw:
      mat.add_(1)
      mat.log_()
      if center:
        mean = mat.mean(dim=0, keepdim=True)
        mat.sub_(mean)
    
    if idx is None:
      result = mat.clone()
    else:
      result = mat[idx, :].clone()

    return result

  def right(self, idx=None, raw=True, center=True):
    """
      given left token, return right frequency/repr
    """
    mat = self.weight.clone()
    if not raw:
      mat.add_(1)
      mat.log_()
      if center:
        mean = mat.mean(dim=0, keepdim=True)
        mat.sub_(mean)
    
    if idx is None:
      result = mat.clone()
    else:
      result = mat[idx, :].clone()

    return result

  def record(
      self,
      left, 
      right, 
      increment_by = 1.0,
    ):
    if isinstance(left, torch.LongTensor) and isinstance(right, torch.LongTensor):
      assert len(left.shape) == len(right.shape) == 1
      assert left.shape == right.shape
      left = left.to(self.weight.device)
      right = right.to(self.weight.device)
      stacked = torch.stack([left, right]) #[2, d]
      unique, inverse, counts = stacked.unique(dim=1, return_counts=True, return_inverse=True)
      unique_left, unique_right = unique.unbind(dim=0)
      new_increment_by = torch.zeros_like(counts, dtype=torch.float)
      new_increment_by[inverse] = increment_by
      self.weight[unique_left, unique_right] += counts * increment_by
    else:
      self.weight[left, right] += increment_by

  def stats(self):
    return {
      "min" : self.weight.min().item(),
      "abs_min" : self.weight.abs().min().item(),
      "max" : self.weight.max().item(),
      "mean" : self.weight.mean().item(),
      "std" : self.weight.std().item(),
    }