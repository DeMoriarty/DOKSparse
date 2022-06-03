typedef struct {
  float value;
  int64_t index;
} pair;

extern "C"
__global__ void topkspspmm(
  const int64_t* __restrict__ pACrowInds, // [m + 1]
  const int64_t* __restrict__ pAColInds,  // [nnz_a]
  const float* __restrict__ pAVals,       // [nnz_b]

  const int64_t* __restrict__ pBCrowInds, // [n + 1]
  const int64_t* __restrict__ pBColInds,  // [nnz_b]
  const float* __restrict__ pBVals,       // [nnz_b]

  const int64_t* __restrict__ pAlpha1,   // [2]
  const int64_t* __restrict__ pAlpha2,   // [2]
  const int64_t* __restrict__ pBeta1,    // [2]
  const int64_t* __restrict__ pBeta2,    // [2]
  const int64_t* __restrict__ pPrime1,   // [2]
  const int64_t* __restrict__ pPrime2,   // [2]

  int64_t* pTopkInds, //[m, nCands]
  float* pTopkVals, //[m, nCands]

  int m, int n, int k
) {
  constexpr int TPB = _TPB_;   // threads per block
  constexpr int MaxNNZPR = _MAXNNZPR_; // max number of nonzero elements per row
  constexpr int StackCap = _STACKCAP_; // stack capacity used for sorting
  constexpr int TileM = _TILEM_; // number of rows from A matrix
  constexpr int TileN = TPB; // number of rows from B matrix to load at each iteration
  
  constexpr int ThreadsPerGroup = 32; //number of threads per thread group
  constexpr int NumGroups = TPB / ThreadsPerGroup; //number of thread groups
  constexpr int GroupTileN = TileN / NumGroups; //number of rows from B matrix per group
  constexpr int NumBuckets = _NUMBUCKETS_; // number of hashmap buckets
  // constexpr int NumBuckets = TileM * MaxNNZPR; // number of hashmap buckets
  
  int tid = threadIdx.x;
  int mStart = blockIdx.x * TileM;
  int gx = tid % ThreadsPerGroup;
  int gy = tid / ThreadsPerGroup;

  extern __shared__ int64_t smemPtr[];
  // SmemTensor2D<int64_t, TileM, MaxNNZPR> smemAColInds(smemPtr);
  // SmemTensor2D<float, TileM, MaxNNZPR> smemAVals(smemPtr);
  SmemTensor2D<int64_t, NumBuckets, 1> smemHashmapKeys(smemPtr);  // [NumBuckets, 1]
  SmemTensor2D<float, NumBuckets, 1> smemHashmapVals(smemHashmapKeys.endPtr); //[NumBuckets, 1]

  SmemTensor1D<int64_t, TPB> smemIndexExchange(smemHashmapVals.endPtr);  //[TPB]
  SmemTensor1D<float, TPB> smemValueExchange(smemIndexExchange.endPtr);  //[TPB]
  
  SmemTensor1D<int, TileM> smemSortTrigger(smemValueExchange.endPtr); //[TileM]
  SmemTensor1D<float, TileM> smemMinValueExchange(smemSortTrigger.endPtr); //[TileM]

  #pragma unroll
  for (int i=0; i<TileM; i++){
    smemSortTrigger.set(i, 0);
  }

  // initialize hashmap
  SmemHashmap<int64_t, float, 1, 1, NumBuckets, TPB> hashmap(
    pAlpha1, pAlpha2,
    pBeta1,  pBeta2, 
    pPrime1, pPrime2,
    smemHashmapKeys,
    smemHashmapVals,
    -1
  );

  // initialize stack
  Stack<pair, StackCap> threadTopkStack[TileM];
  #pragma unroll
  for (int i=0; i<TileM; i++){
    pair empty_pair = { -INFINITY, -1 };
    threadTopkStack[i].fill(empty_pair);
  }
  float threadMinValue[TileM] = { -INFINITY };
  float topkVal[TileM] = { -INFINITY };
  int64_t topkInd[TileM] = { -2 };

  // load tile from A
  #pragma unroll
  for (int i=0; i<TileM; i++){
    int iM = mStart + i;
    if (iM < m){
      int aRowStart = pACrowInds[iM];
      int aRowEnd = pACrowInds[iM + 1];
      int aRowNNZ = aRowEnd - aRowStart;
      #pragma unroll
      for (int j=0; j < div_ru(MaxNNZPR, TPB); j++){
        #pragma unroll
        for (int t=0; t<TPB; t++){
          if (tid == 0){
            int iNZ = j * TPB + t;
            if (iNZ < MaxNNZPR && iNZ < aRowNNZ){
              int64_t iK = pAColInds[aRowStart + iNZ];
              float value[1] = { pAVals[aRowStart + iNZ] };
              int64_t key[1] = { iM * k + iK };
              hashmap.set(key, value);
            }
          }
        }
        // int iNZ = j * TPB + tid;
        // if (iNZ < MaxNNZPR && iNZ < aRowNNZ){
        //   int64_t iK = pAColInds[aRowStart + iNZ];
        //   float value[1] = { pAVals[aRowStart + iNZ] };
        //   int64_t key[1] = { iM * k + iK };
        //   hashmap.set(key, value);
        // }
      }
    }
  }
  __syncthreads();
  
  for (int a = 0; a < div_ru(n, TPB); a++){
    int64_t iN = a * TPB + tid;
    // pTopkInds[(mStart) * TPB + tid] = iN;
    int64_t bRowStart = 0;
    int64_t bRowNNZ = 0;
    float accumulator[TileM] = {0.f};
    if (iN < n){
      bRowStart = pBCrowInds[iN];
      bRowNNZ = pBCrowInds[iN + 1];
    }
    for (int b = 0; b < ThreadsPerGroup; b++){
      int64_t iN2 = a * TPB + gy * ThreadsPerGroup + b;
      int64_t curBRowStart = __shfl_sync(0xffffffff, bRowStart, b);
      int64_t curBRowNNZ = __shfl_sync(0xffffffff, bRowNNZ, b);

      for (int c = 0; c < div_ru(curBRowNNZ, ThreadsPerGroup); c++){
        int iNZ = c * ThreadsPerGroup + gx;
        int64_t iK = -1;
        float bVal = 0.f;
        if (iNZ < curBRowNNZ){
          iK = pBColInds[curBRowStart + iNZ];
          bVal = pBVals[curBRowStart + iNZ];
        }
        #pragma unroll
        for (int d = 0; d < TileM; d++){
          int iM = mStart + d;
          int64_t key[1] = { iM * k + iK };
          float value[1] = { 0.f };
          if (iK != -1){
            bool isFound = hashmap.get(key, value);
          }
          float aVal = value[0];
          float cVal = aVal * bVal;
          warp_sum<float, 32>(cVal);
          // cVal = __shfl_sync(0xffffffff, cVal, 0);
          if (gx == b){
            accumulator[d] += cVal;
          }
        }
      }
    }
    #pragma unroll
    for (int i=0; i<TileM; i++){
      int iM = mStart + i;
      if (iM >= m) continue;
      int candIndex = tid;
      pTopkVals[iM * TPB + candIndex] = accumulator[i];
      pTopkInds[iM * TPB + candIndex] = iN;
    }
    // pTopkInds[(mStart) * TPB + tid] = iN;
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
    // pTopkVals[iM * TPB + candIndex] = topkVal[i];
    // pTopkInds[iM * TPB + candIndex] = topkInd[i];
  }
}