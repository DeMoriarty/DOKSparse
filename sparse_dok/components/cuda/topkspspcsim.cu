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