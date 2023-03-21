import cupy as cp
import torch
import math
import numpy as np

from ..cuda_callable import CudaCallable
from .cuda_kernel import CudaKernel
from ...util import get_absolute_path, dtype2ctype, str2dtype



class ClosedHashmapImplCuda(CudaKernel):
  def __init__(
      self, 
      tpb=256, 
      kpt=1, 
      key_size=1, 
      key_type="long", 
      value_size=1, 
      value_type="float",
    ):
    cu_fnames = [
      "head",
      "smem_tensor",
      "closed_hashmap_impl_class",
      "closed_hashmap_impl",
    ]
    func_names = [
      "closed_hashmap_get",
      "closed_hashmap_set",
      "closed_hashmap_remove",
    ]
    constant_space = {
      "_TPB_": tpb,
      "_KPT_": kpt,
      "_KEYSIZE_": key_size,
      "_VALUESIZE_": value_size,
      "_KEYTYPE_": key_type,
      "_VALUETYPE_": value_type,
    }
    super().__init__(
      cu_files=[get_absolute_path("components", "cuda", f"{name}.cu") for name in cu_fnames],
      save_preview = get_absolute_path("components", "cuda", f"{cu_fnames[-1]}_preview.cu"),
      func_names=func_names,
      constant_space=constant_space,
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
    assert keys.dtype == all_keys.dtype
    assert all_keys.shape[1:] == keys.shape[1:]
    assert all_keys.shape[0] == all_values.shape[0] == all_uuids.shape[0] != 0
    assert keys.device == all_keys.device == all_values.device == all_uuids.device == self.device
    assert alpha1.shape == beta1.shape == prime1.shape == alpha2.shape == beta2.shape == prime2.shape == key_perm.shape == np.prod(all_keys.shape[1:])# == (self.key_size, )
    assert alpha1.dtype == beta1.dtype == prime1.dtype == alpha2.dtype == beta2.dtype == prime2.dtype == key_perm.dtype == torch.long
    assert alpha1.device == beta1.device == prime1.device == alpha2.device == beta2.device == prime2.device == key_perm.device == self.device
    assert keys.is_contiguous() and all_keys.is_contiguous() and all_values.is_contiguous() and all_uuids.is_contiguous()

    key_type = all_keys.dtype
    value_type = all_values.dtype
    key_size = np.prod(all_keys.shape[1:])
    value_size = np.prod(all_values.shape[1:])

    
    n_keys = keys.shape[0]
    n_buckets = all_keys.shape[0]
    if fallback_value is not None:
      assert fallback_value.dtype == value_type
      assert fallback_value.ndim == 1
      assert fallback_value.device == self.device
      assert fallback_value.shape == all_values.shape[1:]
    else:
      fallback_value = torch.zeros(
        value_size,
        device=keys.device,
        dtype=value_type  
      )

    if values is not None:
      assert values.dtype == value_type
      assert values.device == all_keys.device
      # assert values.ndim == 2
      assert values.shape[0] == keys.shape[0]
      assert values.shape[1:] == all_values.shape[1:]
      assert values.is_contiguous()
    else:
      values = torch.empty(
        n_keys,
        *all_values.shape[1:],
        device=keys.device,
        dtype=value_type
      )
      values[:] = fallback_value[None]

    is_found = torch.zeros(n_keys, device=all_keys.device, dtype=torch.bool)

    constants = {
      "_KEYTYPE_": dtype2ctype(key_type),
      "_VALUETYPE_": dtype2ctype(value_type),
      "_KEYSIZE_": key_size,
      "_VALUESIZE_": value_size,
    }
    constants = {**self._default_constants, **constants}

    blocks_per_grid = ( math.ceil(n_keys / (constants["_TPB_"] * constants["_KPT_"])), )
    threads_per_block = (constants["_TPB_"], )
    self.get_function("closed_hashmap_get", constants)(
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
    assert values.shape[0] == keys.shape[0]
    assert all_keys.shape[1:] == keys.shape[1:]
    assert all_values.shape[1:] == values.shape[1:]
    assert keys.dtype == all_keys.dtype
    assert values.dtype == all_values.dtype
    assert all_keys.shape[0] == all_values.shape[0] == all_uuids.shape[0] != 0
    assert keys.device == values.device == all_keys.device == all_values.device == all_uuids.device == self.device
    assert alpha1.shape == beta1.shape == prime1.shape == alpha2.shape == beta2.shape == prime2.shape == key_perm.shape == np.prod(all_keys.shape[1:])
    assert alpha1.dtype == beta1.dtype == prime1.dtype == alpha2.dtype == beta2.dtype == prime2.dtype == key_perm.dtype == torch.long
    assert alpha1.device == beta1.device == prime1.device == alpha2.device == beta2.device == prime2.device == key_perm.device == self.device
    assert keys.is_contiguous() and values.is_contiguous() and all_keys.is_contiguous() and all_values.is_contiguous() and all_uuids.is_contiguous()

    key_type = all_keys.dtype
    value_type = all_values.dtype
    key_size = np.prod(all_keys.shape[1:])
    value_size = np.prod(all_values.shape[1:])

    n_keys = keys.shape[0]
    n_buckets = all_keys.shape[0]
    is_stored = torch.zeros(n_keys, device=self.device, dtype=torch.bool)

    constants = {
      "_KEYTYPE_": dtype2ctype(key_type),
      "_VALUETYPE_": dtype2ctype(value_type),
      "_KEYSIZE_": key_size,
      "_VALUESIZE_": value_size,
    }
    constants = {**self._default_constants, **constants}

    blocks_per_grid = ( math.ceil(n_keys / (constants["_TPB_"] * constants["_KPT_"])), )
    threads_per_block = (constants["_TPB_"], )
    self.get_function("closed_hashmap_set", constants)(
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

  def remove(
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
    ):
    assert all_keys.shape[1:] == keys.shape[1:]# == self.key_size
    assert keys.dtype == all_keys.dtype# == self.torch_key_type
    assert all_keys.shape[0] == all_values.shape[0] == all_uuids.shape[0] != 0
    assert keys.device == all_keys.device == all_values.device == all_uuids.device == self.device
    assert alpha1.shape == beta1.shape == prime1.shape == alpha2.shape == beta2.shape == prime2.shape == key_perm.shape == np.prod(all_keys.shape[1:])
    assert alpha1.dtype == beta1.dtype == prime1.dtype == alpha2.dtype == beta2.dtype == prime2.dtype == key_perm.dtype == torch.long
    assert alpha1.device == beta1.device == prime1.device == alpha2.device == beta2.device == prime2.device == key_perm.device == self.device
    assert keys.is_contiguous() and all_keys.is_contiguous() and all_values.is_contiguous() and all_uuids.is_contiguous()

    key_type = all_keys.dtype
    value_type = all_values.dtype
    key_size = np.prod(all_keys.shape[1:])
    value_size = np.prod(all_values.shape[1:])

    n_keys = keys.shape[0]
    n_buckets = all_keys.shape[0]

    is_removed = torch.zeros(n_keys, device=self.device, dtype=torch.bool)

    constants = {
      "_KEYTYPE_": dtype2ctype(key_type),
      "_VALUETYPE_": dtype2ctype(value_type),
      "_KEYSIZE_": key_size,
      "_VALUESIZE_": value_size,
    }
    constants = {**self._default_constants, **constants}

    blocks_per_grid = ( math.ceil(n_keys / (constants["_TPB_"] * constants["_KPT_"])), )
    threads_per_block = (constants["_TPB_"], )
    self.get_function("closed_hashmap_remove", constants)(
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
        all_keys.data_ptr(), #[n_all_keys, key_size]
        all_values.data_ptr(), #[n_all_keys, value_size]
        all_uuids.data_ptr(), #[n_all_keys]
        is_removed.data_ptr(),
        n_keys, n_buckets,
      ]
    )

    return is_removed