import cupy as cp
import torch
import numpy as np
import math
import os
from torch import Tensor
from itertools import product
import torch
import cupy as cp
from typing import Optional

from ..cuda_callable import CudaCallable

DEVICE = cp.cuda.Device().id

# @cp.memoize(for_each_device=True)
# def cunnex(func_name, func_body):
#   return cp.cuda.compile_with_cache(func_body).get_function(func_name)

class Stream:
    def __init__(self, ptr):
        self.ptr = ptr

class frozendict(dict):
    def __setitem__(self, key, value):
        raise NotImplementedError("frozendict is immutable")

    def __delitem__(self, key):
        raise NotImplementedError("frozendict is immutable")

    def __hash__(self):
        return hash(frozenset(self.items()))

class CudaKernel:
    def __init__(
            self,
            cu_files,
            func_names,
            constant_space: dict = None,
            save_preview = None,
            compiler_backend="nvcc",
            compiler_options=None,
            get_max_dynamic_smem_size=None,
        ):
        """
        A JIT compiled Cuda Kernel.

        Parameters:
            cu_files: list[str | Path]
                paths of .cu files to include.
            
            func_names: list[str]
                name of the cuda global functions that will be compiled

            constant_space: dict[str, str | int | list | tuple]
                each key is a constant (placeholder symbol) in the .cu files that will be replaced with concrete values
                each value is either the value to be replaced with the constant, or a list of values. multiple kernels will be compiled
                for all combinations of all possible values of all constants. 

            save_preview: Optional[str | Path]
                if given, save the complete kernel to the specified path. used for debugging.

            get_max_dynamic_smem_size: Optional[function]
                if provided, sets the max_dynamic_shared_size_bytes of cuda global functions to the value returned by the function. 
                need to provide a function that takes the name of a cuda global function as input, and returns another function.
                the returned function takes a dictionary of placeholder-value pairs as input, and returns the desired maximum dynamic shared memory
                size in bytes. 

            compiler_backend: {'nvcc', 'nvrtc'}
            
            compiler_options: Optional[list | tuple]

        """
        super().__init__()
        self._use_torch_in_cupy_malloc()
        self.device = torch.device(DEVICE)
        self.stream = Stream(torch.cuda.current_stream(device=DEVICE).cuda_stream)
        self._get_max_dynamic_smem_size = get_max_dynamic_smem_size
        
        self._compiler_backend = compiler_backend
        if compiler_options is None:
            compiler_options = (
                '-std=c++17',
                "--device-c",
            )
        self._compiler_options = compiler_options

        self._func_names = func_names
        if constant_space is not None:
            self._default_constants = dict()
            for k, v in constant_space.items():
                if not isinstance(v, (list, tuple)):
                    constant_space[k] = [v]
                self._default_constants[k] = constant_space[k][0]
                
            self._constant_space = constant_space
        else:
            self._constant_space = None
            self._default_constants = None

        kernel = []
        for filepath in cu_files:
            if os.path.exists(filepath):
                with open(filepath, "r") as f:
                    kernel.append(f.read())
            else:
                raise FileNotFoundError(f"{filepath} doesn't exist.")

        self._kernel = "\n".join(kernel)

        if save_preview is not None:
            try:
                with open(save_preview, "w") as f:
                    f.write(self._kernel)
            except:
                print(f"{save_preview} couldn't be created")

        self._kernel_funcs = dict()
        
        if self._constant_space is not None:
            for constant_values in product(*self._constant_space.values()):
                constant_keys = self._constant_space.keys()
                # for k, v in zip(constant_keys, constant_values):
                #     constants[k] = v
                constants = frozendict(zip(constant_keys, constant_values))
                funcs = self._compile_kernel(self._kernel, self._func_names, constants)
                self._kernel_funcs[constants] = funcs
        else:
            funcs = self._compile_kernel(self._kernel, self._func_names)
            self._kernel_funcs["main"] = funcs 
    
    @staticmethod
    def _torch_alloc(size):
        tensor = torch.empty(size, dtype=torch.uint8, device=DEVICE)
        return cp.cuda.MemoryPointer(
            cp.cuda.UnownedMemory(tensor.data_ptr(), size, tensor), 0)

    def _use_torch_in_cupy_malloc(self):
        cp.cuda.set_allocator(self._torch_alloc)

    def _compile_kernel(
            self,
            kernel_str, 
            func_names: list[str], 
            constants: Optional[dict] = None, 
        ):
        funcs = dict()
        if constants is not None:
            for k, v in constants.items():
                kernel_str = kernel_str.replace(str(k), str(v))
        
        for func_name in func_names:
            funcs[func_name] = cp.RawKernel(
                kernel_str,
                func_name,
                backend=self._compiler_backend,
                options=self._compiler_options
            )
            if self._get_max_dynamic_smem_size is not None:
                max_dynamic_smem_size = self._get_max_dynamic_smem_size(func_name)(constants)
                funcs[func_name].max_dynamic_shared_size_bytes = max_dynamic_smem_size
        return funcs

    def get_function(self, func_name, constants=None):
        assert func_name in self._func_names, f"unrecognized function name: {func_name}"
        if self._default_constants is None:
            return self._kernel_funcs["main"][func_name]
        
        if constants is None:
            constants = self._default_constants

        constants = frozendict({**self._default_constants, **constants})
        if constants not in self._kernel_funcs:
            funcs = self._compile_kernel(self._kernel, self._func_names, constants)
            self._kernel_funcs[constants] = funcs
            return funcs[func_name]

        return self._kernel_funcs[constants][func_name]
