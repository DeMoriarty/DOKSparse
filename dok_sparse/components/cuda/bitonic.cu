#define VOLATILE 


CUDA_DEVICE_INLINE float atomicMax(float *address, float val)
{
  int ret = __float_as_int(*address);
  while(val > __int_as_float(ret))
  {
    int old = ret;
    if((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
        break;
  }
  return __int_as_float(ret);
}

CUDA_DEVICE_INLINE unsigned int bfe(
  unsigned int source,
  unsigned int bitIndex
) {
  unsigned int bit;
  asm volatile("bfe.u32 %0, %1, %2, %3;" : "=r"(bit) : "r"((unsigned int) source), "r"(bitIndex), "r"(1));
  return bit;
}

CUDA_DEVICE_INLINE void warp_comparator(
  float &value,
  int64_t &index,
  const int stride,
  const int direction
){
  const float otherValue = __shfl_xor_sync(0xFFFFFFFF, value, stride);
  const int64_t otherIndex = __shfl_xor_sync(0xFFFFFFFF, index, stride);
  if (value != otherValue){
    bool condition = value < otherValue == direction;
    index = condition ? otherIndex : index;
    value = condition ? otherValue : value;
  }
}

template <int TPB>
CUDA_DEVICE_INLINE void block_comparator(
  float &value,
  int64_t &index,
  const int stride,
  const int direction,
  const int laneID,
  VOLATILE float *valueSmem,
  VOLATILE int64_t *indexSmem
){
  __syncthreads();
  valueSmem[laneID] = value;
  indexSmem[laneID] = index;
  __syncthreads();

  float otherValue = valueSmem[laneID ^ stride];
  int64_t otherIndex = indexSmem[laneID ^ stride];

  if (value != otherValue){
    bool condition = value < otherValue == direction;
    value = condition ? otherValue : value;
    index = condition ? otherIndex : index;
  }
  /*
  */
}

CUDA_DEVICE_INLINE void block_comparator_noop(
){
  __syncthreads();
  __syncthreads();
  __syncthreads();
  __syncthreads();
}

CUDA_DEVICE_INLINE void thread_comparator(
  float &value,
  int64_t &index,
  float otherValue,
  int64_t otherIndex,
  const int direction
){
  bool condition = value > otherValue == direction;
  if (value != otherValue && condition){
    value = otherValue;
    index = otherIndex;
  }
}

template <int TPB>
CUDA_DEVICE_INLINE void bitonic_sort_2(
  float &value,
  int64_t &index,
  int laneID
){
  warp_comparator(value, index, 1, bfe(laneID, 1) ^ bfe(laneID, 0));
}

template <int TPB>
CUDA_DEVICE_INLINE void bitonic_sort_4(
  float &value,
  int64_t &index,
  int laneID
){
  bitonic_sort_2<TPB>(value, index, laneID);
  warp_comparator(value, index, 2, bfe(laneID, 2) ^ bfe(laneID, 1));
  warp_comparator(value, index, 1, bfe(laneID, 2) ^ bfe(laneID, 0));
}

template <int TPB>
CUDA_DEVICE_INLINE void bitonic_sort_8(
  float &value,
  int64_t &index,
  int laneID
){
  bitonic_sort_4<TPB>(value, index, laneID);
  warp_comparator(value, index, 4, bfe(laneID, 3) ^ bfe(laneID, 2));
  warp_comparator(value, index, 2, bfe(laneID, 3) ^ bfe(laneID, 1));
  warp_comparator(value, index, 1, bfe(laneID, 3) ^ bfe(laneID, 0));
}

template <int TPB>
CUDA_DEVICE_INLINE void bitonic_sort_16(
  float &value,
  int64_t &index,
  int laneID
){
  bitonic_sort_8<TPB>(value, index, laneID);
  warp_comparator(value, index, 8, bfe(laneID, 4) ^ bfe(laneID, 3));
  warp_comparator(value, index, 4, bfe(laneID, 4) ^ bfe(laneID, 2));
  warp_comparator(value, index, 2, bfe(laneID, 4) ^ bfe(laneID, 1));
  warp_comparator(value, index, 1, bfe(laneID, 4) ^ bfe(laneID, 0));
}

template <int TPB>
CUDA_DEVICE_INLINE void bitonic_sort_32(
  float &value,
  int64_t &index,
  int laneID
){
  bitonic_sort_16<TPB>(value, index, laneID);
  warp_comparator(value, index, 16, bfe(laneID, 5) ^ bfe(laneID, 4));
  warp_comparator(value, index, 8, bfe(laneID, 5) ^ bfe(laneID, 3));
  warp_comparator(value, index, 4, bfe(laneID, 5) ^ bfe(laneID, 2));
  warp_comparator(value, index, 2, bfe(laneID, 5) ^ bfe(laneID, 1));
  warp_comparator(value, index, 1, bfe(laneID, 5) ^ bfe(laneID, 0));
}

template <int TPB>
CUDA_DEVICE_INLINE void bitonic_sort_global_2(
  float &value,
  int64_t &index,
  float otherValue,
  int64_t otherIndex,
  int laneID
) {
  if (TPB - 32 <= threadIdx.x){
    thread_comparator(value, index, otherValue, otherIndex, 0);
    warp_comparator(value, index, 1, !bfe(laneID, 0));
  }
}

template <int TPB>
CUDA_DEVICE_INLINE void bitonic_sort_global_4(
  float &value,
  int64_t &index,
  float otherValue,
  int64_t otherIndex,
  int laneID
) {
  if (TPB - 32 <= threadIdx.x){
    thread_comparator(value, index, otherValue, otherIndex, 0);
    warp_comparator(value, index, 2, !bfe(laneID, 1));
    warp_comparator(value, index, 1, !bfe(laneID, 0));
  }
}

template <int TPB>
CUDA_DEVICE_INLINE void bitonic_sort_global_8(
  float &value,
  int64_t &index,
  float otherValue,
  int64_t otherIndex,
  int laneID
) {
  if (TPB - 32 <= threadIdx.x){
    thread_comparator(value, index, otherValue, otherIndex, 0);
    warp_comparator(value, index, 4, !bfe(laneID, 2));
    warp_comparator(value, index, 2, !bfe(laneID, 1));
    warp_comparator(value, index, 1, !bfe(laneID, 0));
  }
}

template <int TPB>
CUDA_DEVICE_INLINE void bitonic_sort_global_16(
  float &value,
  int64_t &index,
  float otherValue,
  int64_t otherIndex,
  int laneID
) {
  if (TPB - 32 <= threadIdx.x){
    thread_comparator(value, index, otherValue, otherIndex, 0);
    warp_comparator(value, index, 8, !bfe(laneID, 3));
    warp_comparator(value, index, 4, !bfe(laneID, 2));
    warp_comparator(value, index, 2, !bfe(laneID, 1));
    warp_comparator(value, index, 1, !bfe(laneID, 0));
  }
}

template <int TPB>
CUDA_DEVICE_INLINE void bitonic_sort_global_32(
  float &value,
  int64_t &index,
  float otherValue,
  int64_t otherIndex,
  int laneID
) {
  if (TPB - 32 <= threadIdx.x){
    thread_comparator(value, index, otherValue, otherIndex, 0);
    warp_comparator(value, index, 16, !bfe(laneID, 4));
    warp_comparator(value, index, 8, !bfe(laneID, 3));
    warp_comparator(value, index, 4, !bfe(laneID, 2));
    warp_comparator(value, index, 2, !bfe(laneID, 1));
    warp_comparator(value, index, 1, !bfe(laneID, 0));
  }
}


template <int TPB>
CUDA_DEVICE_INLINE void bitonic_sort_64(
  float &value,
  int64_t &index,
  VOLATILE float *valueSmem,
  VOLATILE int64_t *indexSmem,
  int laneID
){
  bitonic_sort_32<TPB>(value, index, laneID);
  block_comparator<TPB>(value, index, 32, bfe(laneID, 6) ^ bfe(laneID, 5), laneID, valueSmem, indexSmem);
  warp_comparator(value, index, 16, bfe(laneID, 6) ^ bfe(laneID, 4));
  warp_comparator(value, index, 8, bfe(laneID, 6) ^ bfe(laneID, 3));
  warp_comparator(value, index, 4, bfe(laneID, 6) ^ bfe(laneID, 2));
  warp_comparator(value, index, 2, bfe(laneID, 6) ^ bfe(laneID, 1));
  warp_comparator(value, index, 1, bfe(laneID, 6) ^ bfe(laneID, 0));
}

template <int TPB>
CUDA_DEVICE_INLINE void bitonic_sort_global_64(
  float &value,
  int64_t &index,
  float otherValue,
  int64_t otherIndex,
  VOLATILE float *valueSmem,
  VOLATILE int64_t *indexSmem,
  int laneID
) {
  if (TPB - 64 <= threadIdx.x){
    thread_comparator(value, index, otherValue, otherIndex, 0);
    block_comparator<TPB>(value, index, 32, !bfe(laneID, 5), laneID, valueSmem, indexSmem);
    warp_comparator(value, index, 16, !bfe(laneID, 4));
    warp_comparator(value, index, 8, !bfe(laneID, 3));
    warp_comparator(value, index, 4, !bfe(laneID, 2));
    warp_comparator(value, index, 2, !bfe(laneID, 1));

    warp_comparator(value, index, 1, !bfe(laneID, 0));
  } else {
    block_comparator_noop();
  }
}


template <int TPB>
CUDA_DEVICE_INLINE void bitonic_sort_128(
  float &value,
  int64_t &index,
  VOLATILE float *valueSmem,
  VOLATILE int64_t *indexSmem,
  int laneID
){
  bitonic_sort_64<TPB>(value, index, valueSmem, indexSmem, laneID);
  block_comparator<TPB>(value, index, 64, bfe(laneID, 7) ^ bfe(laneID, 6), laneID, valueSmem, indexSmem);
  block_comparator<TPB>(value, index, 32, bfe(laneID, 7) ^ bfe(laneID, 5), laneID, valueSmem, indexSmem);
  warp_comparator(value, index, 16, bfe(laneID, 7) ^ bfe(laneID, 4));
  warp_comparator(value, index, 8, bfe(laneID, 7) ^ bfe(laneID, 3));
  warp_comparator(value, index, 4, bfe(laneID, 7) ^ bfe(laneID, 2));
  warp_comparator(value, index, 2, bfe(laneID, 7) ^ bfe(laneID, 1));
  warp_comparator(value, index, 1, bfe(laneID, 7) ^ bfe(laneID, 0));
}

template <int TPB>
CUDA_DEVICE_INLINE void bitonic_sort_global_128(
  float &value,
  int64_t &index,
  float otherValue,
  int64_t otherIndex,
  VOLATILE float *valueSmem,
  VOLATILE int64_t *indexSmem,
  int laneID
) {
  if (TPB - 128 <= threadIdx.x){
    thread_comparator(value, index, otherValue, otherIndex, 0);
    block_comparator<TPB>(value, index, 64, !bfe(laneID, 6), laneID, valueSmem, indexSmem);
    block_comparator<TPB>(value, index, 32, !bfe(laneID, 5), laneID, valueSmem, indexSmem);
    warp_comparator(value, index, 16, !bfe(laneID, 4));
    warp_comparator(value, index, 8, !bfe(laneID, 3));
    warp_comparator(value, index, 4, !bfe(laneID, 2));
    warp_comparator(value, index, 2, !bfe(laneID, 1));
    warp_comparator(value, index, 1, !bfe(laneID, 0));
  } else {
    block_comparator_noop();
    block_comparator_noop();
  }
}

template <int TPB>
CUDA_DEVICE_INLINE void bitonic_sort_256(
  float &value,
  int64_t &index,
  VOLATILE float *valueSmem,
  VOLATILE int64_t *indexSmem,
  int laneID
){
  bitonic_sort_128<TPB>(value, index, valueSmem, indexSmem, laneID);
  block_comparator<TPB>(value, index, 128, bfe(laneID, 8) ^ bfe(laneID, 7), laneID, valueSmem, indexSmem);
  block_comparator<TPB>(value, index, 64, bfe(laneID, 8) ^ bfe(laneID, 6), laneID, valueSmem, indexSmem);
  block_comparator<TPB>(value, index, 32, bfe(laneID, 8) ^ bfe(laneID, 5), laneID, valueSmem, indexSmem);
  warp_comparator(value, index, 16, bfe(laneID, 8) ^ bfe(laneID, 4));
  warp_comparator(value, index, 8, bfe(laneID, 8) ^ bfe(laneID, 3));
  warp_comparator(value, index, 4, bfe(laneID, 8) ^ bfe(laneID, 2));
  warp_comparator(value, index, 2, bfe(laneID, 8) ^ bfe(laneID, 1));
  warp_comparator(value, index, 1, bfe(laneID, 8) ^ bfe(laneID, 0));
}

template <int TPB>
CUDA_DEVICE_INLINE void bitonic_sort_global_256(
  float &value,
  int64_t &index,
  float otherValue,
  int64_t otherIndex,
  VOLATILE float *valueSmem,
  VOLATILE int64_t *indexSmem,
  int laneID
) {
  if (TPB - 256 <= threadIdx.x){
    thread_comparator(value, index, otherValue, otherIndex, 0);
    block_comparator<TPB>(value, index, 128, !bfe(laneID, 7), laneID, valueSmem, indexSmem);
    block_comparator<TPB>(value, index, 64, !bfe(laneID, 6), laneID, valueSmem, indexSmem);
    block_comparator<TPB>(value, index, 32, !bfe(laneID, 5), laneID, valueSmem, indexSmem);
    warp_comparator(value, index, 16, !bfe(laneID, 4));
    warp_comparator(value, index, 8, !bfe(laneID, 3));
    warp_comparator(value, index, 4, !bfe(laneID, 2));
    warp_comparator(value, index, 2, !bfe(laneID, 1));
    warp_comparator(value, index, 1, !bfe(laneID, 0));
  } else {
    block_comparator_noop();
    block_comparator_noop();
    block_comparator_noop();
  }
}


template <int TPB>
CUDA_DEVICE_INLINE void bitonic_sort_512(
  float &value,
  int64_t &index,
  VOLATILE float *valueSmem,
  VOLATILE int64_t *indexSmem,
  int laneID
){
  bitonic_sort_256<TPB>(value, index, valueSmem, indexSmem, laneID);
  block_comparator<TPB>(value, index, 256, bfe(laneID, 9) ^ bfe(laneID, 8), laneID, valueSmem, indexSmem);
  block_comparator<TPB>(value, index, 128, bfe(laneID, 9) ^ bfe(laneID, 7), laneID, valueSmem, indexSmem);
  block_comparator<TPB>(value, index, 64, bfe(laneID, 9) ^ bfe(laneID, 6), laneID, valueSmem, indexSmem);
  block_comparator<TPB>(value, index, 32, bfe(laneID, 9) ^ bfe(laneID, 5), laneID, valueSmem, indexSmem);
  warp_comparator(value, index, 16, bfe(laneID, 9) ^ bfe(laneID, 4));
  warp_comparator(value, index, 8, bfe(laneID, 9) ^ bfe(laneID, 3));
  warp_comparator(value, index, 4, bfe(laneID, 9) ^ bfe(laneID, 2));
  warp_comparator(value, index, 2, bfe(laneID, 9) ^ bfe(laneID, 1));
  warp_comparator(value, index, 1, bfe(laneID, 9) ^ bfe(laneID, 0));
}

template <int TPB>
CUDA_DEVICE_INLINE void bitonic_sort_global_512(
  float &value,
  int64_t &index,
  float otherValue,
  int64_t otherIndex,
  VOLATILE float *valueSmem,
  VOLATILE int64_t *indexSmem,
  int laneID
) {
  if (TPB - 512 <= threadIdx.x){
    thread_comparator(value, index, otherValue, otherIndex, 0);
    block_comparator<TPB>(value, index, 256, !bfe(laneID, 8), laneID, valueSmem, indexSmem);
    block_comparator<TPB>(value, index, 128, !bfe(laneID, 7), laneID, valueSmem, indexSmem);
    block_comparator<TPB>(value, index, 64, !bfe(laneID, 6), laneID, valueSmem, indexSmem);
    block_comparator<TPB>(value, index, 32, !bfe(laneID, 5), laneID, valueSmem, indexSmem);
    warp_comparator(value, index, 16, !bfe(laneID, 4));
    warp_comparator(value, index, 8, !bfe(laneID, 3));
    warp_comparator(value, index, 4, !bfe(laneID, 2));
    warp_comparator(value, index, 2, !bfe(laneID, 1));
    warp_comparator(value, index, 1, !bfe(laneID, 0));
  } else {
    block_comparator_noop();
    block_comparator_noop();
    block_comparator_noop();
    block_comparator_noop();
  }
}


template <int TPB>
CUDA_DEVICE_INLINE void bitonic_sort_1024(
  float &value,
  int64_t &index,
  VOLATILE float *valueSmem,
  VOLATILE int64_t *indexSmem,
  int laneID
){
  bitonic_sort_512<TPB>(value, index, valueSmem, indexSmem, laneID);
  block_comparator<TPB>(value, index, 512, bfe(laneID, 10) ^ bfe(laneID, 9), laneID, valueSmem, indexSmem);
  block_comparator<TPB>(value, index, 256, bfe(laneID, 10) ^ bfe(laneID, 8), laneID, valueSmem, indexSmem);
  block_comparator<TPB>(value, index, 128, bfe(laneID, 10) ^ bfe(laneID, 7), laneID, valueSmem, indexSmem);
  block_comparator<TPB>(value, index, 64, bfe(laneID, 10) ^ bfe(laneID, 6), laneID, valueSmem, indexSmem);
  block_comparator<TPB>(value, index, 32, bfe(laneID, 10) ^ bfe(laneID, 5), laneID, valueSmem, indexSmem);
  warp_comparator(value, index, 16, bfe(laneID, 10) ^ bfe(laneID, 4));
  warp_comparator(value, index, 8, bfe(laneID, 10) ^ bfe(laneID, 3));
  warp_comparator(value, index, 4, bfe(laneID, 10) ^ bfe(laneID, 2));
  warp_comparator(value, index, 2, bfe(laneID, 10) ^ bfe(laneID, 1));
  warp_comparator(value, index, 1, bfe(laneID, 10) ^ bfe(laneID, 0));
}

template <int TPB>
CUDA_DEVICE_INLINE void bitonic_sort_global_1024(
  float &value,
  int64_t &index,
  float otherValue,
  int64_t otherIndex,
  VOLATILE float *valueSmem,
  VOLATILE int64_t *indexSmem,
  int laneID
) {
  if (TPB - 1024 <= threadIdx.x){
    thread_comparator(value, index, otherValue, otherIndex, 0);
    block_comparator<TPB>(value, index, 512, !bfe(laneID, 9), laneID, valueSmem, indexSmem);
    block_comparator<TPB>(value, index, 256, !bfe(laneID, 8), laneID, valueSmem, indexSmem);
    block_comparator<TPB>(value, index, 128, !bfe(laneID, 7), laneID, valueSmem, indexSmem);
    block_comparator<TPB>(value, index, 64, !bfe(laneID, 6), laneID, valueSmem, indexSmem);
    block_comparator<TPB>(value, index, 32, !bfe(laneID, 5), laneID, valueSmem, indexSmem);
    warp_comparator(value, index, 16, !bfe(laneID, 4));
    warp_comparator(value, index, 8, !bfe(laneID, 3));
    warp_comparator(value, index, 4, !bfe(laneID, 2));
    warp_comparator(value, index, 2, !bfe(laneID, 1));
    warp_comparator(value, index, 1, !bfe(laneID, 0));
  } else {
    block_comparator_noop();
    block_comparator_noop();
    block_comparator_noop();
    block_comparator_noop();
    block_comparator_noop();
  }
}

template <int TPB>
CUDA_DEVICE_INLINE void bitonic_sort(
  float &value,
  int64_t &index,
  VOLATILE float *valueSmem,
  VOLATILE int64_t *indexSmem,
  int laneID
){
  if (TPB == 2){
    bitonic_sort_2<TPB>(value, index, laneID);

  } else if (TPB == 4){
    bitonic_sort_4<TPB>(value, index, laneID);

  } else if (TPB == 8){
    bitonic_sort_8<TPB>(value, index, laneID);
    
  } else if (TPB == 16){
    bitonic_sort_16<TPB>(value, index, laneID);
    
  } else if (TPB == 32){
    bitonic_sort_32<TPB>(value, index, laneID);
    
  } else if (TPB == 64){
    bitonic_sort_64<TPB>(value, index, valueSmem, indexSmem, laneID);
    
  } else if (TPB == 128){
    bitonic_sort_128<TPB>(value, index, valueSmem, indexSmem, laneID);
    
  } else if (TPB == 256){
    bitonic_sort_256<TPB>(value, index, valueSmem, indexSmem, laneID);
    
  } else if (TPB == 512){
    bitonic_sort_512<TPB>(value, index, valueSmem, indexSmem, laneID);
    
  } else if (TPB == 1024){
    bitonic_sort_1024<TPB>(value, index, valueSmem, indexSmem, laneID);
    
  }
}

template <int TPB>
CUDA_DEVICE_INLINE void bitonic_sort_global(
  float &value,
  int64_t &index,
  float otherValue,
  int64_t otherIndex,
  VOLATILE float *valueSmem,
  VOLATILE int64_t *indexSmem,
  int laneID
){
  if (TPB == 2){
    bitonic_sort_global_2<TPB>(value, index, otherValue, otherIndex, laneID);

  } else if (TPB == 4){
    bitonic_sort_global_4<TPB>(value, index, otherValue, otherIndex, laneID);

  } else if (TPB == 8){
    bitonic_sort_global_8<TPB>(value, index, otherValue, otherIndex, laneID);
    
  } else if (TPB == 16){
    bitonic_sort_global_16<TPB>(value, index, otherValue, otherIndex, laneID);
    
  } else if (TPB == 32){
    bitonic_sort_global_32<TPB>(value, index, otherValue, otherIndex, laneID);
    
  } else if (TPB == 64){
    bitonic_sort_global_64<TPB>(value, index, otherValue, otherIndex, valueSmem, indexSmem, laneID);
    
  } else if (TPB == 128){
    bitonic_sort_global_128<TPB>(value, index, otherValue, otherIndex, valueSmem, indexSmem, laneID);
    
  } else if (TPB == 256){
    bitonic_sort_global_256<TPB>(value, index, otherValue, otherIndex, valueSmem, indexSmem, laneID);
    
  } else if (TPB == 512){
    bitonic_sort_global_512<TPB>(value, index, otherValue, otherIndex, valueSmem, indexSmem, laneID);
    
  } else if (TPB == 1024){
    bitonic_sort_global_1024<TPB>(value, index, otherValue, otherIndex, valueSmem, indexSmem, laneID);
    
  }
}