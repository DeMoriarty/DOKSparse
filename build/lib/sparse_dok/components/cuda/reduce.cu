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
