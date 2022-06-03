from functools import reduce
from itertools import product
import warnings
import numpy as np
import torch
import torch.nn as nn
import math

class GridSparseTensor(object):
  def __init__(
      self, 
      size,
      indices=None,
      values=None,
      tensor=None,
      grid=None,
      device=None, 
      dtype=None,
      **kwargs
    ):
    # assert tensor is not None or (indices is not None and values is not None)
    self.n_dims = len(size)
    if grid is None:
      grid = tuple(1 for i in range(self.n_dims))

    if tensor is not None:
      assert isinstance(tensor, torch.Tensor) and tensor.layout == torch.sparse_coo
      indices = tensor._indices()
      values = tensor._values()

    elif indices is not None and values is not None:
      pass

    else:
      indices = torch.zeros(self.n_dims, 0, dtype=torch.long)
      values = torch.zeros(0)

    assert self.n_dims == len(grid) == len(size)
    self.grid = grid
    self.size = size
    if device is None:
      self.device = values.device
    else:
      self.device = torch.device(device)
      indices = indices.to(device)
      values = values.to(device)

    if dtype is None:
      self.dtype = values.dtype
    else:
      self.dtype = dtype
      values = values.to(dtype)

    self.index_grid, self.value_grid, self.block_range, self.block_size = self.partition(
      indices, 
      values, 
      self.grid, 
      self.size,
    )

  def partition(self, indices, values, grid, size):
    assert isinstance(grid, (tuple, list, torch.Size))
    assert isinstance(size, (tuple, list, torch.Size))
    n_dims = indices.shape[0]
    assert n_dims == len(grid) == len(size)
    block_size = [math.ceil(i / j) for i,j in zip(size, grid)]
    coords = product(*[range(i) for i in grid])
    grid_indices = np.empty(grid, dtype=object)
    grid_values = np.empty(grid, dtype=object)
    block_range = torch.zeros([*grid, 2, n_dims], dtype=torch.long, device=values.device)
    for coord in coords:
      block_from = torch.tensor([i*j for i, j in zip(coord, block_size)], device=values.device)
      block_to = torch.tensor([(i+1)*j for i, j in zip(coord, block_size)], device=values.device)
      # block_range[coord + (slice(None), 0)] = block_from
      # block_range[coord + (slice(None), 1)] = block_to
      block_range[coord + (0,)] = block_from
      block_range[coord + (1,)] = block_to
      mask = None
      for dim in range(n_dims):
        dim_mask = (block_from[dim] <= indices[dim]) & (indices[dim] < block_to[dim])
        if mask is None:
          mask = dim_mask
        else:
          mask = mask & dim_mask
      grid_indices[coord] = indices[:, mask]# - block_from[:, None]
      grid_values[coord] = values[mask]
    block_range = block_range.to(values.device )
    return grid_indices, grid_values, block_range, block_size

  def reconstruct(self, grid_indices, grid_values):
    indices = torch.cat(grid_indices.flatten().tolist(), dim=-1)
    values = torch.cat(grid_values.flatten().tolist(), dim=-1)
    return indices, values

  def __setitem__(self, selectors, values):
    coords = product(*[range(i) for i in self.grid])
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      selectors = [torch.tensor(selector, device=self.device) for selector in selectors]
      values = torch.tensor(values, device=self.device, dtype=self.dtype)
    broadcasted = torch.broadcast_tensors(*selectors, values)
    selectors = broadcasted[:-1]
    values = broadcasted[-1]

    for coord in coords:
      current_block_range = self.block_range[coord]
      def get_selector_mask(selector, range):
        return (range[0] <= selector) & (selector < range[1])
      selector_masks = [get_selector_mask(selector, current_block_range[:, dim]) for dim, selector in enumerate(selectors)]
      mask = reduce(torch.bitwise_and, selector_masks)
      if mask.sum() == 0:
        continue
      block_selectors = [selector[mask] - current_block_range[0, dim].unsqueeze(-1) for dim, selector in enumerate(selectors)]
      block_indices = self.index_grid[coord] - current_block_range[0][:, None]
      block_values = self.value_grid[coord]
      dense = torch.sparse_coo_tensor(block_indices, block_values, size=self.block_size).to_dense()
      dense[block_selectors] = values[mask]
      sparse = dense.to_sparse()
      new_block_indices = sparse._indices() + current_block_range[0][:, None]
      new_block_values = sparse._values()
      self.index_grid[coord] = new_block_indices
      self.value_grid[coord] = new_block_values

  def __getitem__(self, selectors):
    coords = product(*[range(i) for i in self.grid])
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      selectors = [torch.tensor(selector, device=self.device) for selector in selectors]
    selectors = torch.broadcast_tensors(*selectors)
    broadcasted_shape = torch.broadcast_shapes(*[selector.shape for selector in selectors])

    values = torch.zeros(broadcasted_shape, device=self.device, dtype=self.dtype)
    for coord in coords:
      current_block_range = self.block_range[coord]
      def get_selector_mask(selector, range):
        return (range[0] <= selector) & (selector < range[1])
        
      selector_masks = [get_selector_mask(selector, current_block_range[:, dim]) for dim, selector in enumerate(selectors)]
      mask = reduce(torch.bitwise_and, selector_masks)
      if mask.sum() == 0:
        continue
      block_selectors = [selector[mask] - current_block_range[0, dim].unsqueeze(-1) for dim, selector in enumerate(selectors)]
      block_indices = self.index_grid[coord] - current_block_range[0][:, None]
      block_values = self.value_grid[coord]
      dense = torch.sparse_coo_tensor(block_indices, block_values, size=self.block_size).to_dense()
      values[mask] = dense[block_selectors]
    return values

  def to_sparse(self):
    indices, values = self.reconstruct(self.index_grid, self.value_grid)
    return torch.sparse_coo_tensor(indices, values, size=self.size, device=self.device, dtype=self.dtype)

  def to_dense(self):
    return self.to_sparse().to_dense()
