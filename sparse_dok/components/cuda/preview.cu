#include "cuda_fp16.h"
#include <stdio.h>
// #include "mma"

#define likely(x)      __builtin_expect(!!(x), 1)
#define unlikely(x)    __builtin_expect(!!(x), 0)
#define load(x)        __ldcg(x)
#define store(x, value) __stcs(x, value)
#define div_ru(a, b) (a + b - 1) / b 
#define div_rd(a, b) a / b 
#define VOLATILE
#ifndef INFINITY
#define INFINITY __int_as_float(0x7f800000)
#endif
#define DEBUG

#ifdef DEBUG
#define COMPILER_ASSERT(EXPRESSION)   switch (0) {case 0: case (EXPRESSION):;}
#else
#define COMPILER_ASSERT(EXPRESSION)
#endif

#define CUDA_DEVICE_INLINE __device__ __forceinline__

// #define LAYOUT_C true
// #define LAYOUT_F false
// #define TRANSFORM_N true
// #define TRANSFORM_T false

// typedef bool MATRIX_LAYOUT;
// typedef bool MATRIX_TRANSFORM;
typedef unsigned char uint8_t;
typedef long long ll_t;
typedef unsigned long long ull_t;

typedef struct __builtin_align__(8){
  half x1, x2, x3, x4;
} half4;

typedef struct __builtin_align__(16){
  half x1, x2, x3, x4, x5, x6, x7, x8;
} half8;

typedef struct __builtin_align__(16){
  unsigned char x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16;
} uchar16;

typedef struct __builtin_align__(16){
  char x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16;
} char16;

typedef struct {
  int x;
} Coord1D;

typedef struct {
  int x, y;
} Coord2D;

typedef struct {
  int x, y, z;
} Coord3D;

typedef struct {
  int x, y, z, t;
} Coord4D;

typedef struct {
  int x, y, z, t, u;
} Coord5D;

typedef union {
  int as_int32[1];
  unsigned int as_uint32[1];
  short as_int16[2];
  unsigned short as_uint16[2];
  signed char as_int8[4];
  unsigned char as_uint8[4]; 
  float as_float[1];
  half2 as_half2[1];
  half as_half[2];  
} Data4B;

typedef union {
  long long as_int64[1];
  unsigned long long as_uint64[1];
  int as_int32[2];
  unsigned int as_uint32[2];
  short as_int16[4];
  unsigned short as_uint16[4];
  signed char as_int8[8];
  unsigned char as_uint8[8]; 
  double as_double[1];
  half4 as_half4[1];
  float2 as_float2[1];
  float as_float[2];
  half2 as_half2[2];
  half as_half[4];  
} Data8B;

typedef union {
  uchar16 as_uchar16[1];
  char16 as_char16[1];
  long long as_int64[2];
  unsigned long long as_uint64[2];
  int as_int32[4];
  unsigned int as_uint32[4];
  short as_int16[8];
  unsigned short as_uint16[8];
  signed char as_int8[16];
  unsigned char as_uint8[16];
  half8 as_half8[1]; 
  double as_double[2];
  half4 as_half4[2];
  float2 as_float2[2];
  float as_float[4];
  half2 as_half2[4];
  half as_half[8];  
} Data16B;


template <typename MutexType>
CUDA_DEVICE_INLINE
void mutex_lock_thread(
  MutexType *mutex,
  const MutexType onValue,
  const MutexType offValue
) {
  unsigned int ns = 8;
  unsigned int counter = 0;
  while (atomicCAS(mutex, offValue, onValue) == onValue) {
    __nanosleep(ns);
    counter ++;
    if (counter > 1000) break;
    if (ns < 256) {
      ns *= 2;
    }
  }
}

template <typename MutexType>
CUDA_DEVICE_INLINE
void mutex_unlock_thread(
  MutexType *mutex,
  const MutexType offValue
) {
  __threadfence();
  atomicExch(mutex, offValue);
  __threadfence();
}

CUDA_DEVICE_INLINE
long long atomicCAS(
  ll_t *address,
  ll_t compare,
  ll_t val
){
  ull_t old = atomicCAS(
    reinterpret_cast<ull_t*>(address),
    reinterpret_cast<ull_t&>(compare),
    reinterpret_cast<ull_t&>(val)
  );
  return reinterpret_cast<ll_t&>(old);
}
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
template <
  typename T
>
class SmemTensor0D{
  public:
    VOLATILE T* endPtr;
    VOLATILE T* startPtr;

    CUDA_DEVICE_INLINE
    SmemTensor0D(VOLATILE void* smemPtr)  
        : startPtr(reinterpret_cast<T*>(smemPtr))
        , endPtr(reinterpret_cast<T*>(smemPtr))
    {
    }

    CUDA_DEVICE_INLINE
    T get(){
      return startPtr[0];
    }

    CUDA_DEVICE_INLINE
    T* get_ptr(){
      return startPtr;
    }

    CUDA_DEVICE_INLINE
    void set(T value){
      startPtr[0] = value;
    }

    template <typename U>
    CUDA_DEVICE_INLINE
    U get_reinterpreted(){
      U* newPtr = reinterpret_cast<U*>(startPtr);
      return newPtr[0];
    }

    template <typename U>
    CUDA_DEVICE_INLINE
    void set_reinterpreted(U value){
      U* newPtr = reinterpret_cast<U*>(startPtr);
      newPtr[0] = value;
    }

    template <typename U>
    CUDA_DEVICE_INLINE
    U* get_ptr_reinterpreted(){
      U* newPtr = reinterpret_cast<U*>(startPtr);
      return &newPtr[0];
    }
};

template <
  typename T,
  int ShapeX
>
class SmemTensor1D{
  public:
    VOLATILE T* endPtr;
    VOLATILE T* startPtr;

    CUDA_DEVICE_INLINE
    SmemTensor1D(VOLATILE void* smemPtr)  
        : startPtr(reinterpret_cast<T*>(smemPtr))
        , endPtr(reinterpret_cast<T*>(smemPtr) + shape().x)
    {
    }

    CUDA_DEVICE_INLINE
    T get(int x){
      return startPtr[x];
    }

    CUDA_DEVICE_INLINE
    T* get_ptr(int x){
      return &startPtr[x];
    }

    CUDA_DEVICE_INLINE
    void set(int x, T value){
      startPtr[x] = value;
    }

    template <typename U>
    CUDA_DEVICE_INLINE
    U get_reinterpreted(int x){
      U* newPtr = reinterpret_cast<U*>(startPtr);
      return newPtr[x];
    }

    template <typename U>
    CUDA_DEVICE_INLINE
    void set_reinterpreted(int x, U value){
      U* newPtr = reinterpret_cast<U*>(startPtr);
      newPtr[x] = value;
    }

    template <typename U>
    CUDA_DEVICE_INLINE
    U* get_ptr_reinterpreted(int x){
      U* newPtr = reinterpret_cast<U*>(startPtr);
      return &newPtr[x];
    }

    CUDA_DEVICE_INLINE
    static constexpr Coord1D shape(){
      return { ShapeX };
    }
};

template <
  typename T,
  int ShapeX,
  int ShapeY
>
class SmemTensor2D{
  public:
    VOLATILE T* endPtr;
    VOLATILE T* startPtr;

    CUDA_DEVICE_INLINE
    SmemTensor2D(VOLATILE void* smemPtr)  
        : startPtr(reinterpret_cast<T*>(smemPtr))
        , endPtr(reinterpret_cast<T*>(smemPtr) + shape().x * shape().y)
    {
    }

    CUDA_DEVICE_INLINE
    T get(int x, int y){
      return startPtr[x * stride().x + y];
    }

    CUDA_DEVICE_INLINE
    T* get_ptr(int x, int y){
      return &startPtr[x * stride().x + y];
    }

    CUDA_DEVICE_INLINE
    void set(int x, int y, T value){
      startPtr[x * stride().x + y] = value;
    }

    CUDA_DEVICE_INLINE
    SmemTensor1D<T, ShapeY> get_child(int x){
      SmemTensor1D<T, ShapeY> child(
        &startPtr[x * stride().x]
      );
      return child;
    }

    template <typename U>
    CUDA_DEVICE_INLINE
    U get_reinterpreted(int x, int y){
      U* newPtr = reinterpret_cast<U*>(startPtr);
      return newPtr[
        (x * stride().x) * sizeof(T) / sizeof(U) + 
        y
      ];
    }

    template <typename U>
    CUDA_DEVICE_INLINE
    void set_reinterpreted(int x, int y, U value){
      U* newPtr = reinterpret_cast<U*>(startPtr);
      newPtr[
        (x * stride().x) * sizeof(T) / sizeof(U) + 
        y
      ] = value;
    }

    template <typename U>
    CUDA_DEVICE_INLINE
    U* get_ptr_reinterpreted(int x, int y){
      U* newPtr = reinterpret_cast<U*>(startPtr);
      return &newPtr[
        (x * stride().x) * sizeof(T) / sizeof(U) + 
        y
      ];
    }

    CUDA_DEVICE_INLINE
    static constexpr Coord2D shape(){
      return {ShapeX, ShapeY};
    }

    CUDA_DEVICE_INLINE
    static constexpr Coord1D stride(){
      return {ShapeY};
    }

};

template <
  typename T,
  int ShapeX,
  int ShapeY,
  int ShapeZ
>
class SmemTensor3D{
  public:
    VOLATILE T* endPtr;
    VOLATILE T* startPtr;

    CUDA_DEVICE_INLINE
    SmemTensor3D(VOLATILE void* smemPtr)  
        : startPtr(reinterpret_cast<T*>(smemPtr))
        , endPtr(reinterpret_cast<T*>(smemPtr) + shape().x * shape().y * shape().z)
    {
    }

    CUDA_DEVICE_INLINE
    T get(int x, int y, int z){
      return startPtr[x * stride().x + y * stride().y + z];
    }

    CUDA_DEVICE_INLINE
    T* get_ptr(int x, int y, int z){
      return &startPtr[x * stride().x + y * stride().y + z];
    }

    CUDA_DEVICE_INLINE
    void set(int x, int y, int z, T value){
      startPtr[x * stride().x + y * stride().y + z] = value;
    }

    CUDA_DEVICE_INLINE
    SmemTensor2D<T, ShapeY, ShapeZ> get_child(int x){
      SmemTensor2D<T, ShapeY, ShapeZ> child(
        &startPtr[x * stride().x]
      );
      return child;
    }

    CUDA_DEVICE_INLINE
    SmemTensor1D<T, ShapeZ> get_child(int x, int y){
      SmemTensor1D<T, ShapeZ> child(
        &startPtr[x * stride().x + y * stride().y]
      );
      return child;
    }

    template <typename U>
    CUDA_DEVICE_INLINE
    U get_reinterpreted(int x, int y, int z){
      U* newPtr = reinterpret_cast<U*>(startPtr);
      return newPtr[
        (x * stride().x +  
        y * stride().y) * sizeof(T) / sizeof(U) + 
        z
      ];
    }

    template <typename U>
    CUDA_DEVICE_INLINE
    void set_reinterpreted(int x, int y, int z, U value){
      U* newPtr = reinterpret_cast<U*>(startPtr);
      newPtr[
        (x * stride().x +  
        y * stride().y) * sizeof(T) / sizeof(U) + 
        z
      ] = value;
    }

    template <typename U>
    CUDA_DEVICE_INLINE
    U* get_ptr_reinterpreted(int x, int y, int z){
      U* newPtr = reinterpret_cast<U*>(startPtr);
      return &newPtr[
        (x * stride().x +  
        y * stride().y) * sizeof(T) / sizeof(U) + 
        z
      ];
    }

    CUDA_DEVICE_INLINE
    static constexpr Coord3D shape(){
      return {ShapeX, ShapeY, ShapeZ};
    }

    CUDA_DEVICE_INLINE
    static constexpr Coord2D stride(){
      return {ShapeY * ShapeZ, ShapeZ};
    }

};

template <
  typename T,
  int ShapeX,
  int ShapeY,
  int ShapeZ,
  int ShapeT
>
class SmemTensor4D{
  public:
    VOLATILE T* endPtr;
    VOLATILE T* startPtr;
    // const Coord3D _stride;
    // const Coord4D _shape;

    CUDA_DEVICE_INLINE
    SmemTensor4D(VOLATILE void* smemPtr)  
        : startPtr(reinterpret_cast<T*>(smemPtr))
        , endPtr(&reinterpret_cast<T*>(smemPtr)[shape().x * shape().y * shape().z * shape().t])
    {
    }

    CUDA_DEVICE_INLINE
    T get(int x, int y, int z, int t){
      return startPtr[
        x * stride().x + 
        y * stride().y + 
        z * stride().z +
        t
      ];
    }

    CUDA_DEVICE_INLINE
    T* get_ptr(int x, int y, int z, int t){
      return &startPtr[
        x * stride().x + 
        y * stride().y + 
        z * stride().z +
        t
      ];
    }

    CUDA_DEVICE_INLINE
    void set(int x, int y, int z, int t, T value){
      startPtr[
        x * stride().x + 
        y * stride().y + 
        z * stride().z +
        t
      ] = value;
    }

    CUDA_DEVICE_INLINE
    SmemTensor3D<T, ShapeY, ShapeZ, ShapeT> get_child(int x){
      SmemTensor3D<T, ShapeY, ShapeZ, ShapeT> child(
        &startPtr[x * stride().x]
      );
      return child;
    }

    CUDA_DEVICE_INLINE
    SmemTensor2D<T, ShapeZ, ShapeT> get_child(int x, int y){
      SmemTensor2D<T, ShapeZ, ShapeT> child(
        &startPtr[x * stride().x + y * stride().y]
      );
      return child;
    }

    CUDA_DEVICE_INLINE
    SmemTensor1D<T, ShapeT> get_child(int x, int y, int z){
      SmemTensor1D<T, ShapeT> child(
        &startPtr[x * stride().x + y * stride().y + z * stride().z]
      );
      return child;
    }

    template <typename U>
    CUDA_DEVICE_INLINE
    U get_reinterpreted(int x, int y, int z, int t){
      U* newPtr = reinterpret_cast<U*>(startPtr);
      return newPtr[
        (x * stride().x +  
        y * stride().y +  
        z * stride().z) * sizeof(T) / sizeof(U) + 
        t
      ];
    }

    template <typename U>
    CUDA_DEVICE_INLINE
    void set_reinterpreted(int x, int y, int z, int t, U value){
      U* newPtr = reinterpret_cast<U*>(startPtr);
      newPtr[
        (x * stride().x +  
        y * stride().y +  
        z * stride().z) * sizeof(T) / sizeof(U) + 
        t
      ] = value;
    }

    template <typename U>
    CUDA_DEVICE_INLINE
    U* get_ptr_reinterpreted(int x, int y, int z, int t){
      U* newPtr = reinterpret_cast<U*>(startPtr);
      return &newPtr[
        (x * stride().x +  
        y * stride().y +  
        z * stride().z) * sizeof(T) / sizeof(U) + 
        t
      ];
    }

    CUDA_DEVICE_INLINE
    static constexpr Coord4D shape(){
      return {ShapeX, ShapeY, ShapeZ, ShapeT};
    }

    CUDA_DEVICE_INLINE
    static constexpr Coord3D stride(){
      return {
        ShapeY * ShapeZ * ShapeT, 
        ShapeZ * ShapeT, 
        ShapeT
      };
    }
};
template <typename T, int StackCap>
class Stack{
  private:
    int _stackSize = 0;
    T _stack[StackCap];

  public:
    CUDA_DEVICE_INLINE
    Stack(){
    }

    CUDA_DEVICE_INLINE 
    bool is_empty(){
      return _stackSize <= 0;
    }

    CUDA_DEVICE_INLINE
    bool is_full(){
      return _stackSize >= StackCap - 1;
    }

    CUDA_DEVICE_INLINE
    int size(){
      return _stackSize;
    }

    CUDA_DEVICE_INLINE
    int capacity(){
      return StackCap;
    }

    CUDA_DEVICE_INLINE
    void fill(T item){
      #pragma unroll
      for (int i=0; i < StackCap; i++){
        _stack[i] = item;
      }
    }

    CUDA_DEVICE_INLINE
    void push(T item){
      if (is_full()){
        return;
      } else {
        #pragma unroll
        for (int i = StackCap - 1; i >= 1; i--){
          _stack[i] = _stack[i - 1];
        }
        _stack[0] = item;
        _stackSize ++;
      }
    }

    CUDA_DEVICE_INLINE
    void pop(T &out){
      if (is_empty()){
        return;
      } else {
        out = _stack[0];
        #pragma unroll
        for (int i=0; i<StackCap-1; i++){
          _stack[i] = _stack[i+1];
        }
        _stackSize--;
      }
    }

    CUDA_DEVICE_INLINE
    T pop(){
      T outItem;
      if (!is_empty()) {
        outItem = _stack[0];
        #pragma unroll
        for (int i=0; i<StackCap-1; i++){
          _stack[i] = _stack[i+1];
        }
        _stackSize--;
      }
      return outItem;
    }

};

template <typename T>
CUDA_DEVICE_INLINE
int binary_search_recursive(T *arr, int left, int right, T value)
{
  if (right >= left) {
    int mid = left + (right - left) / 2;

    // If the element is present at the middle
    // itself
    if (arr[mid] == value)
        return mid;

    // If element is smaller than mid, then
    // it can only be present in left subarray
    if (arr[mid] > value)
        return binary_search_recursive(arr, left, mid - 1, value);

    // Else the element can only be present
    // in right subarray
    return binary_search_recursive(arr, mid + 1, right, value);
  }

  // We reach here when element is not
  // present in array
    return -1;
}

template <typename T>
CUDA_DEVICE_INLINE
int binary_search_iterative(T *arr, int size, T value)
{   
    int left = 0;
    int right = size;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] == value)
            return mid;
        if (arr[mid] < value)
            left = mid + 1;
        else
            right = mid - 1;
    }
    return -1;
}

template <typename T> 
CUDA_DEVICE_INLINE
int binary_search_iterative_v2(T* arr, int size, T value)
{
  int low = 0;
  T v = value + 1;
  while (size > 0) {
    int half = size / 2;
    int other_half = size - half;
    int probe = low + half;
    int other_low = low + other_half;
    v = arr[probe];
    size = half;
    low = v < value ? other_low : low;
    if (v == value){
        return probe;
    }
  }
  return -1;
}

template <typename T> 
CUDA_DEVICE_INLINE
int binary_search_iterative_v3(T* arr, int __size, T value)
{
  int low = 0;
  int index = -1;
  #pragma unroll
  for (int size = __size; size > 0; size /= 2){
    int half = size / 2;
    int other_half = size - half;
    int probe = low + half;
    int other_low = low + other_half;
    T v = arr[probe];
    low = v < value ? other_low : low;
    // if (v == value){
    //     return probe;
    // }
    index = v == value ? probe : index;
    
  }
  return index;
}



template <typename T, int BatchSize, int Size> 
CUDA_DEVICE_INLINE
void binary_search_iterative_batched(
  T* arr, // [B, N] 
  T value,
  int indices[BatchSize]
) {
  #pragma unroll
  for (int d=0; d<BatchSize; d++){
    indices[d] = -1;
  }
  int low[BatchSize] = { 0 };
  #pragma unroll
  for (int size = Size; size > 0; size /= 2){
    int half = size / 2;
    int other_half = size - half;
    #pragma unroll
    for (int d=0; d<BatchSize; d++){
      int probe = low[d] + half;
      int other_low = low[d] + other_half;
      T v = arr[d * Size + probe];
      low[d] = v < value ? other_low : low[d];
      // if (v == value){
      //     return probe;
      // }
      indices[d]= v == value ? probe : indices[d];
    }
  } 
}

template <typename T, int N>
CUDA_DEVICE_INLINE
void warp_sum(T &value){
  if (N == 32){
    // warp_sum_32(value);
    // value += __shfl_xor_sync(-1, value, 1);
    // value += __shfl_xor_sync(-1, value, 2);
    // value += __shfl_xor_sync(-1, value, 4);
    // value += __shfl_xor_sync(-1, value, 8);
    // value += __shfl_xor_sync(-1, value, 16);
    value += __shfl_xor_sync(0xffffffff, value, 16);
    value += __shfl_xor_sync(0xffffffff, value, 8);
    value += __shfl_xor_sync(0xffffffff, value, 4);
    value += __shfl_xor_sync(0xffffffff, value, 2);
    value += __shfl_xor_sync(0xffffffff, value, 1);

  } else if (N == 16){
    value += __shfl_xor_sync(0xffffffff, value, 8);
    value += __shfl_xor_sync(0xffffffff, value, 4);
    value += __shfl_xor_sync(0xffffffff, value, 2);
    value += __shfl_xor_sync(0xffffffff, value, 1);

  } else if (N == 8){
    value += __shfl_xor_sync(0xffffffff, value, 4);
    value += __shfl_xor_sync(0xffffffff, value, 2);
    value += __shfl_xor_sync(0xffffffff, value, 1);
    
  } else if (N == 4){
    value += __shfl_xor_sync(0xffffffff, value, 2);
    value += __shfl_xor_sync(0xffffffff, value, 1);
    
  } else if (N == 2){
    value += __shfl_xor_sync(0xffffffff, value, 1);
    
  }
}

template <typename T, int N>
CUDA_DEVICE_INLINE
void warp_sum(unsigned int mask, T &value){
  if (N == 32){
    value += __shfl_xor_sync(mask, value, 16);
    value += __shfl_xor_sync(mask, value, 8);
    value += __shfl_xor_sync(mask, value, 4);
    value += __shfl_xor_sync(mask, value, 2);
    value += __shfl_xor_sync(mask, value, 1);

  } else if (N == 16){
    value += __shfl_xor_sync(mask, value, 8);
    value += __shfl_xor_sync(mask, value, 4);
    value += __shfl_xor_sync(mask, value, 2);
    value += __shfl_xor_sync(mask, value, 1);

  } else if (N == 8){
    value += __shfl_xor_sync(mask, value, 4);
    value += __shfl_xor_sync(mask, value, 2);
    value += __shfl_xor_sync(mask, value, 1);
    
  } else if (N == 4){
    value += __shfl_xor_sync(mask, value, 2);
    value += __shfl_xor_sync(mask, value, 1);
    
  } else if (N == 2){
    value += __shfl_xor_sync(mask, value, 1);
    
  }
}

template <typename T>
CUDA_DEVICE_INLINE
void fast_sum(
    T &value, 
    const int i, 
    T regs[6]
  ){
  const int wx = threadIdx.x % 32;
  const unsigned int mask = 0xffffffff;
  if (i < 32){
    regs[0] = value;
    regs[0] += __shfl_xor_sync(mask, regs[0], 16);
    if ( (wx / 16) == (i % 2) ){
      regs[1] = regs[0];
    }

    if ( i % 2 == 1){
      regs[1] += __shfl_xor_sync(mask, regs[1], 8);
      if (((wx / 8) % 2) == ((i % 4) / 2) ){
        regs[2] = regs[1];
      }
    }

    if (i % 4 == 3){
      regs[2] += __shfl_xor_sync(mask, regs[2], 4);
      if (((wx / 4) % 2) == ((i % 8) / 4) ){
        regs[3] = regs[2];
      }
    }

    if (i % 8 == 7){
      regs[3] += __shfl_xor_sync(mask, regs[3], 2);
      if (((wx / 2) % 2) == ((i % 16) / 8) ){
        regs[4] = regs[3];
      }
    }

    if (i % 16 == 15){
      regs[4] += __shfl_xor_sync(mask, regs[4], 1);
      if ( (wx % 2) == (i / 16) ){
        regs[5] = regs[4];
      }
    }
  } else {
    int srcLane = (wx / 16) + ((wx % 16) / 8) * 2 + ((wx % 8) / 4) * 4 + ((wx % 4) / 2) * 8 + (wx % 2) * 16;
    value = __shfl_sync(mask, regs[5], srcLane);
  }
}

#define SIM_INNER 0
#define SIM_NL1 1
#define SIM_NL2 2
#define SIM_NLP 3

typedef struct {
  float value;
  int64_t index;
} pair;

template <typename T, int n>
CUDA_DEVICE_INLINE
void fill_array(T arr[n], T val){
  #pragma unroll
  for (int i=0; i<n; i++){
    arr[i] = val;
  }
}

extern "C"
__global__ void topkspspcsim(
  const int64_t* __restrict__ pARowStart, // [m + 1]
  const int64_t* __restrict__ pARowNNZ, // [m + 1]
  const int64_t* __restrict__ pAColInds,  // [nnz_a]
  const float* __restrict__ pAVals,       // [nnz_b]

  const int64_t* __restrict__ pBRowStart, // [n]
  const int64_t* __restrict__ pBRowNNZ, // [n]
  const int64_t* __restrict__ pBColInds,  // [nnz_b]
  const float* __restrict__ pBVals,       // [nnz_b]

  int64_t* pTopkInds, //[m, nCands]
  float* pTopkVals, //[m, nCands]

  int m, int n, int k
) {
  constexpr int TPB = _TPB_;   // threads per block
  constexpr int MaxNNZPR = _MAXNNZPR_; // max number of nonzero elements per row
  constexpr int StackCap = _STACKCAP_; // stack capacity used for sorting
  constexpr int TileM = _TILEM_; // number of rows from A matrix
  constexpr int TileN = TPB; // number of rows from B matrix to load at each iteration
  
  constexpr int ThreadsPerGroup = _TPG_; //number of threads per thread group
  constexpr int NumGroups = TPB / ThreadsPerGroup; //number of thread groups
  constexpr int GroupTileN = TileN / NumGroups; //number of rows from B matrix per group
  constexpr int GroupsPerWarp = 32 / ThreadsPerGroup;
  constexpr unsigned int BaseShflMask = (1 << ThreadsPerGroup) - 1; 
  constexpr float P = _P_;
  constexpr int SimType = _SIMTYPE_;
  
  int tid = threadIdx.x;
  int mStart = blockIdx.x * TileM;
  int gx = tid % ThreadsPerGroup;
  int gy = tid / ThreadsPerGroup;

  int group_id_in_warp = (tid % 32) / ThreadsPerGroup;
  unsigned int shflMask = BaseShflMask << (group_id_in_warp * ThreadsPerGroup);

  extern __shared__ int64_t smemPtr[];
  SmemTensor2D<int64_t, TileM, MaxNNZPR> smemAColInds(smemPtr);
  SmemTensor2D<float, TileM, MaxNNZPR> smemAVals(smemAColInds.endPtr);

  SmemTensor1D<int64_t, TPB> smemIndexExchange(smemAVals.endPtr);  //[TPB]
  SmemTensor1D<float, TPB> smemValueExchange(smemIndexExchange.endPtr);  //[TPB]
  
  SmemTensor1D<int, TileM> smemSortTrigger(smemValueExchange.endPtr); //[TileM]
  SmemTensor1D<float, TileM> smemMinValueExchange(smemSortTrigger.endPtr); //[TileM]

  #pragma unroll
  for (int i=0; i<TileM; i++){
    smemSortTrigger.set(i, 0);
  }

  // initialize stack
  Stack<pair, StackCap> threadTopkStack[TileM];
  #pragma unroll
  for (int i=0; i<TileM; i++){
    pair empty_pair = { -INFINITY, -1 };
    threadTopkStack[i].fill(empty_pair);
  }
  float threadMinValue[TileM];
  float topkVal[TileM];
  int64_t topkInd[TileM];
  fill_array<float, TileM>(threadMinValue, -INFINITY);
  fill_array<float, TileM>(topkVal, -INFINITY);
  fill_array<int64_t, TileM>(topkInd, -2);

  // load tile from A
  #pragma unroll
  for (int i=0; i < div_ru(TileM, NumGroups); i++){
    int iMBlock = i * NumGroups + gy;
    int64_t iM = mStart + iMBlock;
    if (iM < m && iMBlock < TileM){
      int64_t aRowStart = pARowStart[iM];
      int64_t aRowNNZ = pARowNNZ[iM];
      #pragma unroll
      for (int j=0; j < div_ru(MaxNNZPR, ThreadsPerGroup); j++){
        int64_t iNZA = j * ThreadsPerGroup + gx;
        if (iNZA < MaxNNZPR){
          if (iNZA < aRowNNZ){
            int64_t iK = pAColInds[aRowStart + iNZA];
            float aVal = pAVals[aRowStart + iNZA];
            smemAColInds.set(iMBlock, iNZA, iK);
            smemAVals.set(iMBlock, iNZA, aVal);
          } else {
            smemAColInds.set(iMBlock, iNZA, 9999999);
            smemAVals.set(iMBlock, iNZA, 0.f);
          }
        }
      }
    }
  }

  __syncthreads();
  int64_t nextBRowStart = 0;
  int64_t nextBRowNNZ = 0;
  if (tid < n){
    nextBRowStart = pBRowStart[tid];
    nextBRowNNZ = pBRowNNZ[tid];
  }
  
  for (int a = 0; a < div_ru(n, TPB); a++){
    float accumulator[TileM]; // (TPB, TileM)
    fill_array<float, TileM>(accumulator, 0.f);
    int64_t iN = a * TPB + tid;

    int64_t bRowStart = nextBRowStart;
    int64_t bRowNNZ = nextBRowNNZ;
    if (a < div_ru(n, TPB) - 1){
      int64_t nextIN = (a+1) * TPB + tid;
      nextBRowStart = 0;
      nextBRowNNZ = 0;
      if (nextIN < n){
        nextBRowStart = pBRowStart[nextIN];
        nextBRowNNZ = pBRowNNZ[nextIN];
      }
    }
    int64_t curBRowStart = __shfl_sync(shflMask, bRowStart, 0, ThreadsPerGroup);
    int64_t curBRowNNZ = __shfl_sync(shflMask, bRowNNZ, 0, ThreadsPerGroup);
    int64_t nextIKB = -1;
    int64_t nextINZB = gx;
    float nextBVal = 0.f;
    if (nextINZB < curBRowNNZ){
      nextIKB = pBColInds[curBRowStart + nextINZB];
      nextBVal = pBVals[curBRowStart + nextINZB];
    }
    float regs[TileM][6];

    // TODO: don't know if this is necessary
    // #pragma unroll
    // for (int d = 0; d < TileM; d++){
    //   #pragma unroll
    //   for (int e = 0; e < 6; e++){
    //     regs[d][e] = 0.f;
    //   }
    // }

    #pragma unroll
    for (int b = 0; b < ThreadsPerGroup; b++){
      int loadItrs = div_ru(curBRowNNZ, ThreadsPerGroup);
      float cVals[TileM];
      fill_array<float, TileM>(cVals, 0.f);
      for (int c = 0; c < loadItrs; c++){
        volatile int64_t iKB = nextIKB;
        volatile float bVal = nextBVal;

        if (c < loadItrs - 1){
          nextINZB = (c+1) * ThreadsPerGroup + gx;
          nextIKB = -1;
          nextBVal = 0.f;
          if (nextINZB < curBRowNNZ){
            nextIKB = pBColInds[curBRowStart + nextINZB];
            nextBVal = pBVals[curBRowStart + nextINZB];
          }
        } else if (b < ThreadsPerGroup - 1) {
          curBRowStart = __shfl_sync(shflMask, bRowStart, b+1, ThreadsPerGroup);
          curBRowNNZ = __shfl_sync(shflMask, bRowNNZ, b+1, ThreadsPerGroup);
          nextINZB = gx;
          nextIKB = -1;
          nextBVal = 0.f;
          if (nextINZB < curBRowNNZ){
            nextIKB = pBColInds[curBRowStart + nextINZB];
            nextBVal = pBVals[curBRowStart + nextINZB];
          }
        }
        #pragma unroll
        for (int d = 0; d < TileM; d++){
          int64_t iM = mStart + d;
          int iNZA;
          #if (_BINSEARCHVER_ == 0)
            iNZA = binary_search_recursive<int64_t>(smemAColInds.get_child(d).startPtr, 0, MaxNNZPR, iKB);
          #elif (_BINSEARCHVER_ == 1)
            iNZA = binary_search_iterative<int64_t>(smemAColInds.get_child(d).startPtr, MaxNNZPR, iKB);
          #elif (_BINSEARCHVER_ == 2)
            iNZA = binary_search_iterative_v2<int64_t>(smemAColInds.get_child(d).startPtr, MaxNNZPR, iKB);
          #elif (_BINSEARCHVER_ == 3)
            iNZA = binary_search_iterative_v3<int64_t>(smemAColInds.get_child(d).startPtr, MaxNNZPR, iKB);
          #endif

          float aVal = iNZA == -1 ? 0.f : smemAVals.get(d, iNZA);
          if (SimType == SIM_INNER){
            cVals[d] += aVal * bVal;

          } else if (SimType == SIM_NL1){
            cVals[d] += fabsf(aVal - bVal);

          } else if (SimType == SIM_NL2){
            float dif = aVal - bVal;
            cVals[d] += dif * dif;
          
          } else if (SimType == SIM_NLP){
            float dif = aVal - bVal;
            if (P == 0.f){
              cVals[d] += dif == 0.f ? 0.f : 1.f;
            } else {
              // if (P % 2.f == 1.f){
              //   dif = fabsf(dif);
              // }
              dif = fabsf(dif);
              cVals[d] += powf(dif, P);
            }
          }
        }
      }
      #pragma unroll
      for (int d=0; d<TileM; d++){
        // fast_sum<float>(cVals[d], b, regs[d]);
        warp_sum<float, ThreadsPerGroup>(shflMask, cVals[d]);
        if (gx == b){
          accumulator[d] = cVals[d];
        }
      }
    }
    // #pragma unroll
    // for (int d=0; d<TileM; d++){
    //   fast_sum<float>(accumulator[d], ThreadsPerGroup, regs[d]);
    // }


    // push index value pair into stack
    __syncthreads();
    pair oldPairs[TileM];
    #pragma unroll
    for (int i = 0; i < TileM; i++){
      oldPairs[i] = { -INFINITY, -3 };
      if (threadTopkStack[i].is_full()){
        threadTopkStack[i].pop(oldPairs[i]);
        if (oldPairs[i].value > threadMinValue[i]){
          smemSortTrigger.set(i, 1);
        }
      }

      pair newPair;
      if (iN < n){
        newPair = { accumulator[i],  iN};
      } else {
        newPair = { -INFINITY, -4 };
      }
      
      if (accumulator[i] > threadMinValue[i]){
        threadTopkStack[i].push(newPair);
      }
    }
    __syncthreads();

    // sort if necessary
    #pragma unroll
    for (int i=0; i<TileM; i++){
      if (smemSortTrigger.get(i) > 0){
        __syncthreads();
        bitonic_sort<TPB>(oldPairs[i].value, oldPairs[i].index, 
                          smemValueExchange.startPtr, smemIndexExchange.startPtr, 
                          tid);

        bitonic_sort_global<TPB>(topkVal[i], topkInd[i], 
                                oldPairs[i].value, oldPairs[i].index, 
                                smemValueExchange.startPtr, smemIndexExchange.startPtr,
                                tid);
        __syncthreads();
        if (tid == TPB - 1){
          smemMinValueExchange.set(i, topkVal[i]);
        }
        __syncthreads();
        threadMinValue[i] = smemMinValueExchange.get(i);
      }
    }
    __syncthreads();
  }
  // sort the remaining items in stack
  #pragma unroll
  for (int i=0; i<TileM; i++){
    smemSortTrigger.set(i, 0);
    __syncthreads();

    #pragma unroll
    for (int j=0; j<StackCap; j++){
      pair oldPair = { -INFINITY, -5 };
      if (!threadTopkStack[i].is_empty()){
         threadTopkStack[i].pop(oldPair);
         if (oldPair.value > threadMinValue[i]){
           smemSortTrigger.set(i, 1);
         }
      }
      __syncthreads();

      if (smemSortTrigger.get(i) > 0){
        __syncthreads();
        bitonic_sort<TPB>(oldPair.value, oldPair.index, 
                          smemValueExchange.startPtr, smemIndexExchange.startPtr, 
                          tid);

        bitonic_sort_global<TPB>(topkVal[i], topkInd[i], 
                                oldPair.value, oldPair.index, 
                                smemValueExchange.startPtr, smemIndexExchange.startPtr,
                                tid);
        __syncthreads();
        smemSortTrigger.set(i, 0);
        if (tid == TPB - 1){
          smemMinValueExchange.set(i, topkVal[i]);
        }
        __syncthreads();
        threadMinValue[i] = smemMinValueExchange.get(i);
      }
      __syncthreads();
    }
  }

  // write results back
  #pragma unroll
  for (int i=0; i<TileM; i++){
    int iM = mStart + i;
    if (iM >= m) continue;
    int candIndex = tid;
    pTopkVals[iM * TPB + candIndex] = topkVal[i];
    pTopkInds[iM * TPB + candIndex] = topkInd[i];
  }
}