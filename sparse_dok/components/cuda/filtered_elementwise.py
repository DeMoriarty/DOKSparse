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

class FilteredElementwise(CudaCallable):
    def __init__(
            self,
            tpb=256,
        ):
        super().__init__()
        self.tpb = tpb

        cu_fnames = [
            "head",
            "smem_tensor"
            "closed_hashmap_impl_class"
            "filtered_elementwise",
        ]

        kernel = []
        for fname in cu_fnames:
            with open(get_absolute_path("components", "cuda", f"{fname}.cu"), "r") as f:
                kernel.append(f.read())
        self.kernel = "\n".join(kernel)

        with open(get_absolute_path("components", "cuda", f"{cu_fnames[-1]}_preview.cu"), "w") as f:
            f.write(self.kernel)

        self.kernel = (self.kernel
            .replace("_TPB_", str(self.tpb))
        )

        self.fn = cp.RawKernel(
            self.kernel,
            "filtered_elementwise_1x2",
            backend="nvcc",
            options=(
                '-std=c++17',
                "--device-c",
            )
        )

    def __call__(self):
        pass