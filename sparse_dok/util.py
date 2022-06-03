import numpy as np
import torch
from datetime import datetime
from pathlib import Path
import math
import psutil
from os import path

def sliding_window(array, window_size, start_at=0, max_count=None):
  if max_count is None:
    max_count = len(array)
  start = start_at + 1 - window_size + 1
  
  sub_windows = (
    start +
    # expand_dims are used to convert a 1D array to 2D array.
    np.expand_dims(np.arange(window_size), 0) +
    np.expand_dims(np.arange(max_count + 1), 0).T
  ) % len(array)
  return torch.tensor(array[sub_windows], dtype=torch.long)

class TableLogger:
  def __init__(self, log_dir, col_labels, col_types=None, col_lengths=None):
    self.log_dir = Path(log_dir)
    now = datetime.now()
    self.file_path = self.log_dir / f"{now.date()}_{now.hour}_{now.minute}.log"
    self.n_cols = len(col_labels)
    if col_types is None:
      col_types = [str for i in range(self.n_cols)]
    
    if col_lengths is None:
      # col_lengths = [15 for i in range(self.n_cols)]
      col_lengths = [max(3, len(label)) for label in col_labels]
    assert len(col_labels) == len(col_types) == len(col_lengths)

    self.col_labels = col_labels
    self.col_types = col_types
    self.col_lengths = col_lengths
    self.format_str = [f"{{:<{col_lengths[i]}.{col_lengths[i]}}}" for i in range(self.n_cols)]
    self.format_str = f"| {' | '.join(self.format_str)} |\n"
    # self.title_line = self.format_str.format(*self.col_labels)
    # self.dash_line = self.format_str.format(*["-"*i for i in self.col_lengths])

    self.add_row(self.col_labels, remember_types=False)
    self.add_row(["-"*i for i in self.col_lengths], remember_types=False)

  def add_row(self, row_items, remember_types=True):
    assert len(row_items) == self.n_cols
    if remember_types:
      self.col_types = [type(item) for item in row_items]
    line = self.format_str.format(*[str(item) for item in row_items])
    with open(self.file_path, "a") as f:
      f.write(line)

  def read_row(self, row):
    items = row.split("|")[1:-1]
    if len(items) != self.n_cols:
      return None
    
    # items = [item.strip() for item in items]
    items = [items[i].strip() for i in range(self.n_cols)]
    items = [self.col_types[i](items[i]) for i in range(self.n_cols)]
    return items

  def read_table(self):
    with open(self.file_path, "r") as f:
      rows = f.read().split("\n")[2:]
    row_items = [self.read_row(row) for row in rows]
    row_items = [i for i in row_items if i is not None]
    return row_items

def str2dtype(s):
  if isinstance(s, torch.dtype):
    return s
  elif isinstance(s, str):
    if s in ["bool"]:
      return torch.bool
    elif s in ["uint8", "char"]:
      return torch.uint8
    elif s in ["int8", "byte"]:
      return torch.int8
    elif s in ["int16", "short"]:
      return torch.int16
    elif s in ["int32", "int"]:
      return torch.int32
    elif s in ["int 64", "long"]:
      return torch.int64
    elif s in ["float16", "half"]:
      return torch.float16
    elif s in ["bfloat16"]:
      return torch.bfloat16
    elif s in ["float32", "float", ]:
      return torch.float32
    elif s in ["float64", "double"]:
      return torch.float64
    else:
      raise ValueError(f"unrecognized type string {s}")

  else:
    raise RuntimeError

def dtype2ctype(dtype):
  dtype = str2dtype(dtype)
  if dtype == torch.bool:
    return "bool"
  elif dtype == torch.uint8:
    return "uint8_t"
  elif dtype == torch.int8:
    return "int8_t"
  elif dtype == torch.int16:
    return "int16_t"
  elif dtype == torch.int32:
    return "int32_t"
  elif dtype == torch.int64:
    return "int64_t"
  elif dtype == torch.float16:
    return "half"
  elif dtype == torch.bfloat16:
    return "bfloat16"
  elif dtype == torch.float32:
    return "float"
  elif dtype == torch.float64:
    return "double"

def next_power_of_2(n):
    count = 0
    # First n in the below
    # condition is for the
    # case where n is 0
    if (n and not(n & (n - 1))):
        return n
     
    while( n != 0):
        n >>= 1
        count += 1
     
    return 1 << count

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

def expand_tensor(tensor, dim=0, fill=0, by=1.0):
  shape = list(tensor.shape)
  n_dims = len(shape)
  assert n_dims > 0, "tensor needs to have at least 1 dimension"
  if isinstance(by, float):
    shape[dim] = math.ceil(shape[dim] * by)
  elif isinstance(by, int):
    shape[dim] = by
  
  new_tensor = torch.zeros(shape, dtype=tensor.dtype, device=tensor.device)
  new_tensor.fill_(fill)
  new_tensor = torch.cat([tensor, new_tensor], dim=dim)
  return new_tensor

def batch_allclose(a, b, tol=1e-6):
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
    return ((a - b).abs() >= tol).sum(dim=-1) == 0

def check_available_ram(device="cpu"):
  """
  Returns available RAM on target device
  args:
    device:     str or torch.device
  """
  if isinstance(device, str):
    device = torch.device(device)
  elif isinstance(device, torch.device):
    device = device
  else:
    raise RuntimeError("`device` must be str or torch.device")
  
  if device.type == "cpu":
    return psutil.virtual_memory().available
  else:
    total = torch.cuda.get_device_properties(device).total_memory
    used = torch.cuda.memory_allocated(device)
    return total - used

def will_it_fit(size, device="cpu", safe_mode=True):
  """
  Returns True if an array of given byte size fits in target device.
  if self.safe_mode = False, this function simply compares the given byte size with the remaining RAM on target device. This option is faster, 
      but it doesn't take memory fragmentation into account. So it will still be possible to run out of memory.
  if self.safe_mode = True, it will try to allocate a tensor with the given size. if allocation fails, return False. 
      This option is recommended when the other option fails because of OOM.
  
  args:
      size:       int
      device:     str or torch.device
      safe_mode:  bool
  returns:
      result:     bool
  """
  if safe_mode:
    try:
      torch.empty(size, device=device, dtype=torch.uint8)
    except:
      return False
    return True
  else:
    return check_available_ram(device) >= size

def find_optimal_splits(n, get_required_memory, device="cpu", safe_mode=True):
  """
  Find an optimal number of split for `n`, such that `get_required_memory(math.ceil(n / n_split))` fits in target device's RAM.
  get_required_memory should be a fucntion that receives `math.ceil(n/n_split)` and returns the required memory in bytes.
  args:
    n:                      int
    get_required_memory:    function
    device:                 str or torch.device
    safe_mode:              bool
  returns:
    n_splits:               int
  """
  splits = 1
  sub_n = n
  break_next = False
  while True:
    if break_next:
      break
    if splits > n:
      splits = n
      break_next = True
    sub_n = math.ceil(n / splits)
    required_memory = get_required_memory(sub_n)
    if will_it_fit(required_memory, device):
      break
    else:
      splits *= 2
      continue
  return splits

def get_absolute_path(*relative_path):
  relative_path = path.join(*relative_path)
  return path.join(path.dirname(__file__), relative_path)

def merge(candidate_pairs, selected_pair_idx, replacement):
  """
    candidate_pairs : [2, n_seqs, new_seq_len]
    selected_pair_idx : [n_seqs]
    replacement : [n_seqs]
  """
  n_seqs = selected_pair_idx.shape[0]
  new_seq_len = candidate_pairs.shape[2]
  assert candidate_pairs.shape == (2, n_seqs, new_seq_len)
  assert selected_pair_idx.shape == (n_seqs,)
  assert replacement.shape == (n_seqs,)
  arange = torch.arange(new_seq_len)[None]
  # print(arange.shape)
  # print(selected_pair_idx[:, None].shape)
  # print(candidate_pairs.shape)
  # print(candidate_pairs.shape)
  # print(selected_pair_idx.shape)
  # print((arange < selected_pair_idx).shape )

  merged_seq = candidate_pairs[0] * (arange < selected_pair_idx[:, None])
  merged_seq += candidate_pairs[1] * (arange > selected_pair_idx[:, None])
  merged_seq += replacement[:, None] * (arange == selected_pair_idx[:, None])
  return merged_seq
