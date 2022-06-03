import cupy as cp
import torch
import math
from torch import Tensor

from ..cuda_callable import CudaCallable
from ...util import get_absolute_path, dtype2ctype, str2dtype

class ClosedHashmapImplCuda(CudaCallable):
  def __init__(
      self, 
      tpb=256, 
      kpt=1, 
      key_size=1, 
      key_type="long", 
      value_size=1, 
      value_type="float",
    ):
    super().__init__()
    assert 32 <= tpb <= 1024
    assert kpt >= 1
    assert key_size >= 1 
    assert value_size >= 1

    self.tpb = tpb
    self.kpt = kpt
    self.torch_key_type = str2dtype(key_type)
    self.torch_value_type = str2dtype(value_type)
    self.key_type = dtype2ctype(key_type)
    self.value_type = dtype2ctype(value_type)
    self.key_size = key_size
    self.value_size = value_size

    cu_files = [
      "head.cu",
      "smem_tensor.cu",
      "closed_hashmap_impl.cu",
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
      .replace("_KPT_", str(self.kpt))
      .replace("_KEYSIZE_", str(self.key_size))
      .replace("_VALUESIZE_", str(self.value_size))
      .replace("_KEYTYPE_", str(self.key_type))
      .replace("_VALUETYPE_", str(self.value_type))
    )

    self._get_fn = cp.RawKernel(
      self.kernel,
      "closed_hashmap_get",
      backend="nvcc",
      options=(
        '-std=c++17',
        "--device-c",
      )
    )

    self._set_fn = cp.RawKernel(
      self.kernel,
      "closed_hashmap_set",
      backend="nvcc",
      options=(
        '-std=c++17',
        "--device-c",
      )
    )
    # self.fn.max_dynamic_shared_size_bytes = self.smem_size
  
  def get(
      self,
      prime1,
      prime2,
      alpha1,
      alpha2,
      beta1,
      beta2,
      key_perm ,
      keys, #[n_keys, key_size]
      all_keys, #[n_all_keys, key_size]
      all_values, #[n_all_keys, value_size]
      all_uuids, #[n_all_keys]
      values = None, #[n_keys, value_size]
      fallback_value = None, #[value_size]
    ):
    assert keys.ndim == all_keys.ndim == all_values.ndim == 2
    assert all_keys.shape[1] == keys.shape[1] == self.key_size
    assert all_values.shape[1] == self.value_size
    assert keys.dtype == all_keys.dtype == self.torch_key_type
    assert all_values.dtype == self.torch_value_type
    assert all_keys.shape[0] == all_values.shape[0] == all_uuids.shape[0] != 0
    assert keys.device == all_keys.device == all_values.device == all_uuids.device == self.device
    assert alpha1.shape == beta1.shape == prime1.shape == alpha2.shape == beta2.shape == prime2.shape == (self.key_size, )
    assert alpha1.dtype == beta1.dtype == prime1.dtype == alpha2.dtype == beta2.dtype == prime2.dtype == torch.long
    assert alpha1.device == beta1.device == prime1.device == alpha2.device == beta2.device == prime2.device == self.device
    assert keys.is_contiguous() and all_keys.is_contiguous() and all_values.is_contiguous() and all_uuids.is_contiguous()
    assert key_perm.shape == alpha1.shape
    assert key_perm.device == alpha1.device
    assert key_perm.dtype == torch.long

    
    n_keys = keys.shape[0]
    n_buckets = all_keys.shape[0]
    if fallback_value is not None:
      assert fallback_value.dtype == self.torch_value_type
      assert fallback_value.ndim == 1
      assert fallback_value.device == self.device
      assert fallback_value.shape[0] == self.value_size
    else:
      fallback_value = torch.zeros(
        self.value_size,
        device=keys.device,
        dtype=self.torch_value_type  
      )

    if values is not None:
      assert values.dtype == self.torch_value_type
      assert values.device == self.device
      assert values.ndim == 2
      assert values.shape[0] == keys.shape[0]
      assert values.shape[1] == self.value_size
      assert values.is_contiguous()
    else:
      values = torch.empty(
        n_keys,
        self.value_size,
        device=keys.device,
        dtype=self.torch_value_type
      )
      values[:] = fallback_value[None]

    is_found = torch.zeros(n_keys, device=self.device, dtype=torch.bool)

    blocks_per_grid = ( math.ceil(n_keys / (self.tpb * self.kpt)), )
    threads_per_block = (self.tpb, )
    self._get_fn(
      grid=blocks_per_grid,
      block=threads_per_block,
      args=[
        prime1.data_ptr(),
        prime2.data_ptr(),
        alpha1.data_ptr(), 
        alpha2.data_ptr(),
        beta1.data_ptr(),
        beta2.data_ptr(),
        key_perm.data_ptr(),
        keys.data_ptr(), #[n_keys, key_size]
        values.data_ptr(), #[n_keys, value_size]
        all_keys.data_ptr(), #[n_all_keys, key_size]
        all_values.data_ptr(), #[n_all_keys, value_size]
        all_uuids.data_ptr(), #[n_all_keys]
        fallback_value.data_ptr(), #[value_size]
        is_found.data_ptr(),
        n_keys, n_buckets,
      ]
    )

    return values, is_found

  def set(
      self,
      prime1,
      prime2,
      alpha1,
      alpha2,
      beta1,
      beta2,
      key_perm,
      keys, #[n_keys, key_size]
      values, #[n_keys, value_size]
      all_keys, #[n_all_keys, key_size]
      all_values, #[n_all_keys, value_size]
      all_uuids, #[n_all_keys]
    ):
    assert keys.ndim == values.ndim == all_keys.ndim == all_values.ndim == 2
    assert values.shape[0] == keys.shape[0]
    assert all_keys.shape[1] == keys.shape[1] == self.key_size
    assert all_values.shape[1] == values.shape[1] == self.value_size
    assert keys.dtype == all_keys.dtype == self.torch_key_type
    assert values.dtype == all_values.dtype == self.torch_value_type
    assert all_keys.shape[0] == all_values.shape[0] == all_uuids.shape[0] != 0
    assert keys.device == values.device == all_keys.device == all_values.device == all_uuids.device == self.device
    assert alpha1.shape == beta1.shape == prime1.shape == alpha2.shape == beta2.shape == prime2.shape == (self.key_size, )
    assert alpha1.dtype == beta1.dtype == prime1.dtype == alpha2.dtype == beta2.dtype == prime2.dtype == torch.long
    assert alpha1.device == beta1.device == prime1.device == alpha2.device == beta2.device == prime2.device == self.device
    assert key_perm.shape == alpha1.shape
    assert key_perm.device == alpha1.device
    assert key_perm.dtype == torch.long
    assert keys.is_contiguous() and values.is_contiguous() and all_keys.is_contiguous() and all_values.is_contiguous() and all_uuids.is_contiguous()

    n_keys = keys.shape[0]
    n_buckets = all_keys.shape[0]
    is_stored = torch.zeros(n_keys, device=self.device, dtype=torch.bool)

    blocks_per_grid = ( math.ceil(n_keys / (self.tpb * self.kpt)), )
    threads_per_block = (self.tpb, )
    self._set_fn(
      grid=blocks_per_grid,
      block=threads_per_block,
      args=[
        prime1.data_ptr(),
        prime2.data_ptr(),
        alpha1.data_ptr(), 
        alpha2.data_ptr(),
        beta1.data_ptr(),
        beta2.data_ptr(),
        key_perm.data_ptr(),
        keys.data_ptr(), #[n_keys, key_size]
        values.data_ptr(), #[n_keys, value_size]
        all_keys.data_ptr(), #[n_all_keys, key_size]
        all_values.data_ptr(), #[n_all_keys, value_size]
        all_uuids.data_ptr(), #[n_all_keys]
        is_stored.data_ptr(),
        n_keys, n_buckets,
      ]
    )
    return is_stored