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