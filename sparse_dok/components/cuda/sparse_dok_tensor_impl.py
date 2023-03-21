import cupy as cp
import torch
import numpy as np
import math
from torch import Tensor
from typing import Union
from itertools import product

# import sympy
from torchtimer import ProfilingTimer
timer = ProfilingTimer()

from .cuda_kernel import CudaKernel
from ...util import get_absolute_path, get_tensor_accessor_argument_array, shape_to_stride, dtype2ctype, unique_first_occurrence
from ..cuda_closed_hashmap import CudaClosedHashmap

class SparseDOKTensorImplCuda(CudaKernel):
    def __init__(
            self,
            tpb=32,
            ept=1,
            selector_ndims=[1],
            value_types = ["long", "float", "double", "int"],
            ndims=[1,2],
        ):
        cu_fnames = [
            "head",
            "smem_tensor",
            "reduce",
            "tensor_accessor",
            "closed_hashmap_impl_class",
            "sparse_dok_tensor_impl",
        ]

        func_names = [
            "sparse_dok_count_items",
            "sparse_dok_zero_items",
            "sparse_dok_get_items",
            "sparse_dok_set_items_sparse_v1",
            "sparse_dok_set_items_sparse",
            "sparse_dok_set_items_dense",
        ]
        super().__init__(
            cu_files = [get_absolute_path("components", "cuda", f"{name}.cu") for name in cu_fnames],
            save_preview = get_absolute_path("components", "cuda", f"{cu_fnames[-1]}_preview.cu"),
            func_names=func_names,
            constant_space={
                "_SELECTORNDIM_": selector_ndims,
                "_VALUETYPE_": value_types,
                "_NDIM_": ndims,
                "_TPB_": tpb,
                "_EPT_": ept,
            },
        )

    def count_items(self, hashmap: CudaClosedHashmap, selectors: list[Tensor]):
        timer.start("asserts")
        assert all(isinstance(selector, Tensor) for selector in selectors), "all selectors must be Long Tensors"
        assert all(selector.dtype == torch.long for selector in selectors), "all selectors must be Long Tensors"
        assert all(selector.device == hashmap.device for selector in selectors), "all selectors must be on the same cuda device as the SparseDOKTensor"
        assert all(selector.shape == selectors[0].shape for selector in selectors), "all selectors must be broadcasted to the same shape"
        assert len(selectors) == hashmap.key_shape[0], "number of selectors must be equal to the number of dimensions"
        ndim = len(selectors)
        selector_ndim = selectors[0].ndim

        timer.stop_and_start("asserts", "prepare constants")
        constants = self._default_constants.copy()
        constants["_SELECTORNDIM_"] = selector_ndim
        constants["_NDIM_"] = ndim
        constants["_VALUETYPE_"] = dtype2ctype(hashmap.value_type)
        timer.stop_and_start("prepare constants", "get func")
        func = self.get_function("sparse_dok_count_items", constants)
        # if frozen_constants not in self.kernel_funcs:
        #     funcs = self._compile_kernel(self.kernel, constants, func_names=self.func_names)
        #     self.kernel_funcs[frozen_constants] = funcs
        # func = self.kernel_funcs[frozen_constants]["sparse_dok_count_items"]
        timer.stop_and_start("get func", "selector stride")

        selector_shape = selectors[0].shape
        selector_stride = shape_to_stride(selector_shape)
        timer.stop_and_start("selector stride", "n selector elements")
        n_selector_elements = np.prod(selector_shape)
        timer.stop_and_start("n selector elements", "selector element accessor args")
        selector_element_accessor_args = get_tensor_accessor_argument_array(selectors[0], stride=selector_stride)
        timer.stop_and_start("selector element accessor args", "selector element accessor args to device")
        selector_element_accessor_args = selector_element_accessor_args.to(hashmap.device)
        timer.stop_and_start("selector element accessor args to device", "tensor accessor args")
        tensor_accessor_args = get_tensor_accessor_argument_array(*selectors).to(hashmap.device)

        timer.stop_and_start("tensor accessor args", "kernel")
        counts = torch.zeros(1, device=hashmap.device, dtype=torch.long)


        blocks_per_grid = (math.ceil(n_selector_elements / (constants["_TPB_"] * constants["_EPT_"]) ), )
        threads_per_block = (constants["_TPB_"], )
        func(
            grid=blocks_per_grid,
            block=threads_per_block,
            args=[
                hashmap._kernel_args(as_list=False).data_ptr(),
                tensor_accessor_args.data_ptr(),
                selector_element_accessor_args.data_ptr(),
                counts.data_ptr(),
                n_selector_elements,
            ]
        )
        timer.stop("kernel")
        return counts

    def zero_items(self, hashmap: CudaClosedHashmap, selectors: list[Tensor]):
        assert all(isinstance(selector, Tensor) for selector in selectors), "all selectors must be Long Tensors"
        assert all(selector.dtype == torch.long for selector in selectors), "all selectors must be Long Tensors"
        assert all(selector.device == hashmap.device for selector in selectors), "all selectors must be on the same cuda device as the SparseDOKTensor"
        assert all(selector.shape == selectors[0].shape for selector in selectors), "all selectors must be broadcasted to the same shape"
        assert len(selectors) == hashmap.key_shape[0], "number of selectors must be equal to the number of dimensions"
        ndim = len(selectors)
        selector_ndim = selectors[0].ndim


        selector_shape = selectors[0].shape
        selector_stride = shape_to_stride(selector_shape)
        n_selector_elements = np.prod(selector_shape)
        selector_element_accessor_args = get_tensor_accessor_argument_array(selectors[0], stride=selector_stride)
        selector_element_accessor_args = selector_element_accessor_args.to(hashmap.device)
        tensor_accessor_args = get_tensor_accessor_argument_array(*selectors).to(hashmap.device)

        constants = self._default_constants.copy()
        constants["_SELECTORNDIM_"] = selector_ndim
        constants["_NDIM_"] = ndim
        constants["_VALUETYPE_"] = dtype2ctype(hashmap.value_type)

        n_removed = torch.zeros(1, device=hashmap.device, dtype=torch.long)
        blocks_per_grid = (math.ceil(n_selector_elements / (constants["_TPB_"] * constants["_EPT_"]) ), )
        threads_per_block = (constants["_TPB_"], )
        self.get_function("sparse_dok_zero_items", constants)(
            grid=blocks_per_grid,
            block=threads_per_block,
            args=[
                hashmap._kernel_args(as_list=False).data_ptr(),
                tensor_accessor_args.data_ptr(),
                selector_element_accessor_args.data_ptr(),
                n_removed.data_ptr(),
                n_selector_elements,
            ]
        )
        return n_removed

    def get_items(self, hashmap: CudaClosedHashmap, selectors: list[Tensor], n=None):
        timer.start("asserts", "get_items")
        assert all(isinstance(selector, Tensor) for selector in selectors), "all selectors must be Long Tensors"
        assert all(selector.dtype == torch.long for selector in selectors), "all selectors must be Long Tensors"
        assert all(selector.device == hashmap.device for selector in selectors), "all selectors must be on the same cuda device as the SparseDOKTensor"
        assert all(selector.shape == selectors[0].shape for selector in selectors), "all selectors must be broadcasted to the same shape"
        assert len(selectors) == hashmap.key_shape[0], "number of selectors must be equal to the number of dimensions"
        ndim = len(selectors)
        selector_ndim = selectors[0].ndim

        timer.stop_and_start("asserts", "prepare constants", "get_items")
        constants = self._default_constants.copy()
        constants["_SELECTORNDIM_"] = selector_ndim
        constants["_NDIM_"] = ndim
        constants["_VALUETYPE_"] = dtype2ctype(hashmap.value_type)
        timer.stop_and_start("prepare constants", "get func", "get_items")
        # if frozen_constants not in self.kernel_funcs:
        #     funcs = self._compile_kernel(self.kernel, constants, func_names=self.func_names)
        #     self.kernel_funcs[frozen_constants] = funcs
        # count_items_func = self.kernel_funcs[frozen_constants]["sparse_dok_count_items"]
        # get_items_func = self.kernel_funcs[frozen_constants]["sparse_dok_get_items"]
        count_items_func = self.get_function("sparse_dok_count_items", constants)
        get_items_func = self.get_function("sparse_dok_get_items", constants)
        timer.stop_and_start("get func", "selector stride", "get_items")

        selector_shape = selectors[0].shape
        selector_stride = shape_to_stride(selector_shape)
        timer.stop_and_start("selector stride", "n selector elements", "get_items")
        n_selector_elements = np.prod(selector_shape)
        timer.stop_and_start("n selector elements", "selector element accessor args", "get_items")
        selector_element_accessor_args = get_tensor_accessor_argument_array(selectors[0], stride=selector_stride)
        timer.stop_and_start("selector element accessor args", "selector element accessor args to device", "get_items")
        selector_element_accessor_args = selector_element_accessor_args.to(hashmap.device)
        timer.stop_and_start("selector element accessor args to device", "tensor accessor args", "get_items")
        tensor_accessor_args = get_tensor_accessor_argument_array(*selectors).to(hashmap.device)

        hashmap_args = hashmap._kernel_args(as_list=False)

        timer.stop_and_start("tensor accessor args", "count items kernel", "get_items")
        if n is None:
            n = torch.zeros(1, device=hashmap.device, dtype=torch.long)[0]
            blocks_per_grid = (math.ceil(n_selector_elements / (constants["_TPB_"] * constants["_EPT_"]) ), )
            threads_per_block = (constants["_TPB_"], )
            count_items_func(
                grid=blocks_per_grid,
                block=threads_per_block,
                args=[
                    hashmap_args.data_ptr(),
                    tensor_accessor_args.data_ptr(),
                    selector_element_accessor_args.data_ptr(),
                    n.data_ptr(),
                    n_selector_elements,
                ]
            )
        elif isinstance(n, torch.Tensor):
            assert n.ndim == 0
            n = n.item()

        timer.stop_and_start("count items kernel", "get items kernel prep", "get_items")
        out_hashmap = CudaClosedHashmap(n_buckets=int(n * 2.0), device=hashmap.device)
        # mock_key = torch.zeros(1, selector_ndim, device=hashmap.device, dtype=torch.long)
        # mock_value = torch.zeros(1, 1, device=hashmap.device, dtype=hashmap.value_type) 
        # out_hashmap[mock_key] = mock_value
        timer.stop_and_start("get items kernel prep", "initialize out hashmap", "get_items")
        out_hashmap._keys = torch.zeros(out_hashmap.n_buckets, selector_ndim, device=hashmap.device, dtype=torch.long)
        out_hashmap._values = torch.zeros(out_hashmap.n_buckets, 1, device=hashmap.device, dtype=hashmap.value_type)
        # del out_hashmap[mock_key]

        timer.stop_and_start("initialize out hashmap", "get items kernel", "get_items")
        blocks_per_grid = (math.ceil(n_selector_elements / (constants["_TPB_"] * constants["_EPT_"]) ), )
        threads_per_block = (constants["_TPB_"], )
        get_items_func(
            grid=blocks_per_grid,
            block=threads_per_block,
            args=[
                hashmap_args.data_ptr(),
                tensor_accessor_args.data_ptr(),
                selector_element_accessor_args.data_ptr(),
                out_hashmap._kernel_args(as_list=False).data_ptr(),
                n_selector_elements,
            ]
        )
        out_hashmap._n_elements += n
        timer.stop("get items kernel", "get_items")
        return out_hashmap

    def set_items_sparse(self, dest_hashmap: CudaClosedHashmap, src: Union[CudaClosedHashmap, Tensor], selectors: list[Tensor]):
        # raise NotImplementedError("this shit isn't working")
        assert dest_hashmap.device == src.device
        assert all(isinstance(selector, Tensor) for selector in selectors), "all selectors must be Long Tensors"
        assert all(selector.dtype == torch.long for selector in selectors), "all selectors must be Long Tensors"
        assert all(selector.device == dest_hashmap.device for selector in selectors), "all selectors must be on the same cuda device as the SparseDOKTensor"
        assert all(selector.shape == selectors[0].shape for selector in selectors), "all selectors must be broadcasted to the same shape"
        assert len(selectors) == dest_hashmap.key_shape[0], "number of selectors must match the number of dimensions of the destination tensor"
        if isinstance(src, CudaClosedHashmap):
            assert selectors[0].ndim == src.key_shape[0]
            src_indices = src.keys().T.contiguous()
            src_values = src.values().flatten()

        elif isinstance(src, Tensor) and src.is_sparse:
            assert selectors[0].ndim == src._indices().shape[0]
            if src.layout != torch.sparse_coo:
                src = src.to_sparse_coo()
            src_indices = src.indices()
            src_values = src.values()
        else:
            raise TypeError(f"src needs to be either a CudaClosedHashmap, or a sparse torch Tensor")

        # ufo_mask = unique_first_occurrence(src_indices, dim=-1)
        # src_indices = src_indices[:, ufo_mask]
        # src_values = src_values[ufo_mask]
        # print("0", src_indices.unique(dim=-1, return_counts=True)[1].max())

        ndim = len(selectors)
        selector_ndim = selectors[0].ndim
        device = dest_hashmap.device
        n_src_elements = len(src_values)
        ###### 
        stacked_selectors = torch.stack(selectors, dim=0).reshape(ndim, -1)
        # print(stacked_selectors.shape, stacked_selectors.unique(dim=-1).shape)

        selector_shape = selectors[0].shape
        selector_stride = shape_to_stride(selector_shape)
        n_selector_elements = np.prod(selector_shape)
        selector_element_accessor_args = get_tensor_accessor_argument_array(selectors[0], stride=selector_stride)
        selector_element_accessor_args = selector_element_accessor_args.to(device)
        tensor_accessor_args = get_tensor_accessor_argument_array(*selectors).to(device)

        # _, is_found = dest_hashmap.get(src_indices.T)
        # n_new_elements = (~is_found).sum().item()

        constants = self._default_constants.copy()
        constants["_SELECTORNDIM_"] = selector_ndim
        constants["_NDIM_"] = ndim
        constants["_VALUETYPE_"] = dtype2ctype(dest_hashmap.value_type)
        
        blocks_per_grid = (math.ceil(n_selector_elements / (constants["_TPB_"] * constants["_EPT_"]) ), )
        threads_per_block = (constants["_TPB_"], )
        n_removed = torch.zeros(size=(), device=device, dtype=torch.long)
        # print("before zero", self.count_items(dest_hashmap, selectors))
        # print(dest_hashmap.uuid().unique(return_counts=True)[1].max())
        # print("before", (dest_hashmap._uuid == -3).sum()  )
        self.get_function("sparse_dok_zero_items", constants)(
            grid=blocks_per_grid,
            block=threads_per_block,
            args=[
                dest_hashmap._kernel_args(as_list=False).data_ptr(),
                tensor_accessor_args.data_ptr(),
                selector_element_accessor_args.data_ptr(),
                n_removed.data_ptr(),
                n_selector_elements,
            ]
        )
        # torch.cuda.synchronize()
        dest_hashmap._n_elements -= n_removed

        # print("after", (dest_hashmap._uuid == -3).sum(), n_removed)
        # print("after zero", self.count_items(dest_hashmap, selectors), n_removed)
        # print(dest_hashmap.uuid().unique(return_counts=True)[1].max())

        # if (dest_hashmap.n_elements + n_src_elements) > dest_hashmap.n_buckets * dest_hashmap._rehash_threshold:
        # print("1", dest_hashmap.keys().unique(dim=0, return_counts=True)[1].max())
        if True:
            new_n_buckets = math.ceil((dest_hashmap.n_elements + n_src_elements) * dest_hashmap._rehash_factor * 2)
            print("rehashing...", dest_hashmap.n_buckets, new_n_buckets)
            dest_hashmap.rehash(new_n_buckets)
        # print("2", dest_hashmap.keys().unique(dim=0, return_counts=True)[1].max())
        # print("after rehash", (dest_hashmap._uuid == -3).sum())

        blocks_per_grid = (math.ceil(n_src_elements / (constants["_TPB_"] * constants["_EPT_"]) ), )
        threads_per_block = (constants["_TPB_"], )
        torch.cuda.synchronize()
        self.get_function("sparse_dok_set_items_sparse_v1", constants)(
            grid=blocks_per_grid,
            block=threads_per_block,
            args=[
                src_indices.data_ptr(),
                src_values.data_ptr(),
                dest_hashmap._kernel_args(as_list=False).data_ptr(),
                tensor_accessor_args.data_ptr(),
                selector_element_accessor_args.data_ptr(),
                n_selector_elements,
                n_src_elements,
            ]
        )
        # print("3", dest_hashmap.keys().unique(dim=0, return_counts=True)[1].max())
        dest_hashmap._n_elements += n_src_elements

    def set_items_sparse_v2(self, dest_hashmap: CudaClosedHashmap, src: Union[CudaClosedHashmap, Tensor], selectors: list[Tensor]):
        raise NotImplementedError("this isn't working")
        assert dest_hashmap.device == src.device
        assert all(isinstance(selector, Tensor) for selector in selectors), "all selectors must be Long Tensors"
        assert all(selector.dtype == torch.long for selector in selectors), "all selectors must be Long Tensors"
        assert all(selector.device == dest_hashmap.device for selector in selectors), "all selectors must be on the same cuda device as the SparseDOKTensor"
        assert all(selector.shape == selectors[0].shape for selector in selectors), "all selectors must be broadcasted to the same shape"
        assert len(selectors) == dest_hashmap.key_shape[0], "number of selectors must match the number of dimensions of the destination tensor"
        if isinstance(src, CudaClosedHashmap):
            assert selectors[0].ndim == src.key_shape[0]
            # src_indices = src.keys().T.contiguous()
            # src_values = src.values().flatten()

        elif isinstance(src, Tensor) and src.is_sparse:
            assert selectors[0].ndim == src._indices().shape[0]
            if src.layout != torch.sparse_coo:
                src = src.to_sparse_coo()
            src_indices = src.indices().T.contiguous()
            src_values = src.values().unsqueeze(-1)
            src = CudaClosedHashmap( int(len(src_values) * 2.0) )
            src[src_indices] = src_values
        else:
            raise TypeError(f"src needs to be either a CudaClosedHashmap, or a sparse torch Tensor")

        ndim = len(selectors)
        selector_ndim = selectors[0].ndim
        device = dest_hashmap.device
        n_src_elements = len(src_values)

        selector_shape = selectors[0].shape
        selector_stride = shape_to_stride(selector_shape)
        n_selector_elements = np.prod(selector_shape)
        selector_element_accessor_args = get_tensor_accessor_argument_array(selectors[0], stride=selector_stride)
        selector_element_accessor_args = selector_element_accessor_args.to(device)
        tensor_accessor_args = get_tensor_accessor_argument_array(*selectors).to(device)

        # _, is_found = dest_hashmap.get(src_indices.T)
        # n_new_elements = (~is_found).sum().item()

        constants = self._default_constants.copy()
        constants["_SELECTORNDIM_"] = selector_ndim
        constants["_NDIM_"] = ndim
        constants["_VALUETYPE_"] = dtype2ctype(dest_hashmap.value_type)
        
        blocks_per_grid = (math.ceil(n_selector_elements / (constants["_TPB_"] * constants["_EPT_"]) ), )
        threads_per_block = (constants["_TPB_"], )
        # n_removed = torch.zeros(size=(), device=device, dtype=torch.long)

        if (dest_hashmap.n_elements + n_selector_elements) > dest_hashmap.n_buckets * dest_hashmap._rehash_threshold:
            new_n_buckets = math.ceil((dest_hashmap.n_elements + n_selector_elements) * dest_hashmap._rehash_factor)
            print("rehashing")
            dest_hashmap.rehash(new_n_buckets * 2)

        blocks_per_grid = (math.ceil(n_selector_elements / (constants["_TPB_"] * constants["_EPT_"]) ), )
        threads_per_block = (constants["_TPB_"], )

        self.get_function("sparse_dok_set_items_sparse", constants)(
            grid=blocks_per_grid,
            block=threads_per_block,
            args=[
                src._kernel_args(as_list=False).data_ptr(),
                dest_hashmap._kernel_args(as_list=False).data_ptr(),
                tensor_accessor_args.data_ptr(),
                selector_element_accessor_args.data_ptr(),
                dest_hashmap._n_elements.data_ptr(),
                n_selector_elements,
            ]
        )
        # dest_hashmap._n_elements += n_src_elements

    def set_items_dense(self, dest_hashmap: CudaClosedHashmap, src: Tensor, selectors: list[Tensor]):
        assert dest_hashmap.device == src.device
        assert all(isinstance(selector, Tensor) for selector in selectors), "all selectors must be Long Tensors"
        assert all(selector.dtype == torch.long for selector in selectors), "all selectors must be Long Tensors"
        assert all(selector.device == dest_hashmap.device for selector in selectors), "all selectors must be on the same cuda device as the SparseDOKTensor"
        assert all(selector.shape == selectors[0].shape for selector in selectors), "all selectors must be broadcasted to the same shape"
        assert len(selectors) == dest_hashmap.key_shape[0], "number of selectors must match the number of dimensions of the destination tensor"
        assert src.shape == selectors[0].shape

        ndim = len(selectors)
        selector_ndim = selectors[0].ndim
        device = dest_hashmap.device

        selector_shape = selectors[0].shape
        selector_stride = shape_to_stride(selector_shape)
        n_selector_elements = np.prod(selector_shape)
        selector_element_accessor_args = get_tensor_accessor_argument_array(selectors[0], stride=selector_stride)
        selector_element_accessor_args = selector_element_accessor_args.to(device)
        tensor_accessor_args = get_tensor_accessor_argument_array(*selectors, src).to(device)

        constants = self._default_constants.copy()
        constants["_SELECTORNDIM_"] = selector_ndim
        constants["_NDIM_"] = ndim
        constants["_VALUETYPE_"] = dtype2ctype(dest_hashmap.value_type)
        blocks_per_grid = (math.ceil(n_selector_elements / (constants["_TPB_"] * constants["_EPT_"]) ), )
        threads_per_block = (constants["_TPB_"], )
        
        n_existing_elements = torch.zeros(1, device=device, dtype=torch.long)

        blocks_per_grid = (math.ceil(n_selector_elements / (constants["_TPB_"] * constants["_EPT_"]) ), )
        threads_per_block = (constants["_TPB_"], )
        self.get_function("sparse_dok_count_items", constants)(
            grid=blocks_per_grid,
            block=threads_per_block,
            args=[
                dest_hashmap._kernel_args(as_list=False).data_ptr(),
                tensor_accessor_args.data_ptr(),
                selector_element_accessor_args.data_ptr(),
                n_existing_elements.data_ptr(),
                n_selector_elements,
            ]
        )

        n_new_elements = n_selector_elements - n_existing_elements.item()
        if (dest_hashmap.n_elements + n_selector_elements) > dest_hashmap.n_buckets * dest_hashmap._rehash_threshold:
            dest_hashmap.rehash( math.ceil((dest_hashmap.n_elements + n_selector_elements) * dest_hashmap._rehash_factor) )

        self.get_function("sparse_dok_set_items_dense", constants)(
            grid=blocks_per_grid,
            block=threads_per_block,
            args=[
                dest_hashmap._kernel_args(as_list=False).data_ptr(),
                tensor_accessor_args.data_ptr(),
                selector_element_accessor_args.data_ptr(),
                n_selector_elements,
            ]
        )
        dest_hashmap._n_elements += n_new_elements