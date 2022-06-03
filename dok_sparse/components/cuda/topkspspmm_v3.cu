typedef struct {
  float value;
  int64_t index;
} pair;

extern "C"
__global__ void topkspspmm(
  const int64_t* __restrict__ pACrowInds, // [m + 1]
  const int64_t* __restrict__ pAColInds,  // [nnz_a]
  const float* __restrict__ pAVals,       // [nnz_a]

  const int64_t* __restrict__ pAlpha1,   // [2]
  const int64_t* __restrict__ pAlpha2,   // [2]
  const int64_t* __restrict__ pBeta1,    // [2]
  const int64_t* __restrict__ pBeta2,    // [2]
  const int64_t* __restrict__ pPrime1,   // [2]
  const int64_t* __restrict__ pPrime2,   // [2]
  const int64_t* __restrict__ pHeadNode, // [n_buckets_b]
  const int64_t* __restrict__ pNextNode, // [nnz_b]
  const int64_t* __restrict__ pUUID,     // [nnz_b]
  const float* __restrict__ pBVals,      // [nnz_b]

  int64_t* pTopkInds, //[m, nCands]
  float* pTopkVals, //[m, nCands]

  int m, int n, int k,
  int nBuckets
) {
  constexpr int TPB = _TPB_;   // threads per block
  constexpr int ARowsPerBlock = _ARPB_; // A rows per block
  constexpr int MaxNNZPerRow = _MAXNNZPR_; // max number of nonzeros per row
  constexpr int StackCap = _STACKCAP_; // stack capacity
  
  int tid = threadIdx.x;
  const int mStart = blockIdx.x * ARowsPerBlock;

  extern __shared__ int64_t smemPtr[];
  SmemTensor2D<int64_t, ARowsPerBlock, MaxNNZPerRow> smemAColInds(smemPtr);       //[ARowsPerBlock, maxnnzpr]
  SmemTensor2D<float, ARowsPerBlock, MaxNNZPerRow> smemAVals(reinterpret_cast<float*>(smemAColInds.endPtr));  //[ARowsPerBlock, maxnnzpr]
  
  SmemTensor1D<float, TPB> smemValueExchange(reinterpret_cast<float*>(smemAVals.endPtr));  //[TPB]
  SmemTensor1D<int64_t, TPB> smemIndexExchange(reinterpret_cast<int64_t*>(smemValueExchange.endPtr));  //[TPB]
  SmemTensor1D<int, ARowsPerBlock> smemSortTrigger(reinterpret_cast<int*>(smemIndexExchange.endPtr)); //[ARowsPerBlock]
  SmemTensor1D<float, ARowsPerBlock> smemMinValueExchange(reinterpret_cast<float*>(smemSortTrigger.endPtr)); //[ARowsPerBlock]

  #pragma unroll
  for (int i=0; i<ARowsPerBlock; i++){
    smemSortTrigger.set(i, 0);
  }
    
  // initialize hashmap
  Hashmap<2, int64_t> hashmap(pAlpha1, pAlpha2,
                              pBeta1, pBeta2,
                              pPrime1, pPrime2,
                              pHeadNode, pNextNode,
                              pUUID, nBuckets);
  
  // load rows inds and values from A
  int aRowNNZ[ARowsPerBlock] = {0};
  #pragma unroll
  for (int i=0; i<ARowsPerBlock; i++){
    int iM = mStart + i;
    if (iM < m){
      int rowStart = pACrowInds[iM];
      int rowEnd = pACrowInds[iM+1];
      aRowNNZ[i] = rowEnd - rowStart;
      #pragma unroll
      for (int j=0; j< div_ru(MaxNNZPerRow, TPB); j++){
        int nonzeroInd = j * TPB + tid;
        if (nonzeroInd < MaxNNZPerRow && nonzeroInd < aRowNNZ[i]){
          int64_t colInd = pAColInds[rowStart + nonzeroInd];
          float val = pAVals[rowStart + nonzeroInd];
          smemAColInds.set(i, nonzeroInd, colInd);
          smemAVals.set(i, nonzeroInd, val);
        }
      }
    }
  }

  // initialize stack
  Stack<pair, StackCap> threadTopkStack[ARowsPerBlock];
  #pragma unroll
  for (int i=0; i<ARowsPerBlock; i++){
    pair empty_pair = { -INFINITY, -1 };
    threadTopkStack[i].fill(empty_pair);
  }
  float threadMinValue[ARowsPerBlock] = { -INFINITY };
  float topkVal[ARowsPerBlock] = { -INFINITY };
  int64_t topkInd[ARowsPerBlock] = { -2 };
  
  // start iterating over rows of B
  // each thread is responsible for one row
  // no need to worry about uncoalesced memory access, because it's random access anyways. 

  __syncthreads();
  for (int itr=0; itr < div_ru(n, TPB); itr ++){
    int64_t iN = itr * TPB + tid;
    pair newPairs[ARowsPerBlock];
    #pragma unroll
    for (int i=0; i<ARowsPerBlock; i++){
      int64_t iM = mStart + i;
      if (iM < m && iN < n){
        newPairs[i] = { 0.f, iN };
      } else {
        newPairs[i] = { -INFINITY, -1 };
      }
    }
    // __syncthreads();

    // load elements from A and B and matmul
    #pragma unroll
    for (int j=0; j<MaxNNZPerRow; j++){
      int64_t keys[ARowsPerBlock][2];
      #pragma unroll
      for (int i=0; i<ARowsPerBlock; i++){
        int64_t iK = smemAColInds.get(i, j);
        keys[i][0] = iN;
        keys[i][1] = iK;
      }
      int64_t hashCode[ARowsPerBlock];
      int64_t UUID[ARowsPerBlock];
      #pragma unroll
      for (int i=0; i<ARowsPerBlock; i++){
        hashCode[i] = hashmap.get_hash_subkey<1>( { keys[i][0] }, {0} );
        UUID[i] = hashmap.get_uuid(keys[i]);
      }

      int64_t lastVisitedNode[ARowsPerBlock] = { -1 };
      bool isFound[ARowsPerBlock] = { false };
      hashmap.find_nodes<ARowsPerBlock>(keys, lastVisitedNode, isFound, ARowsPerBlock);
      
      float bVals[ARowsPerBlock] = { 0.f };
      #pragma unroll
      for (int i=0; i<ARowsPerBlock; i++){
        int64_t iM = mStart + i;
        int64_t iK = smemAColInds.get(i, j);
        if (iM >= m || iN >= n || j >= aRowNNZ[i]) continue;
        float aVal = smemAVals.get(i, j);
        if (isFound[i]){
          bVals[i] = pBVals[lastVisitedNode[i]];
        }
      }

      #pragma unroll
      for (int i=0; i<ARowsPerBlock; i++){
        int64_t iM = mStart + i;
        if (iM >= m || iN >= n || j >= aRowNNZ[i]) continue;
        float aVal = smemAVals.get(i, j);
        float bVal = bVals[i];
        newPairs[i].value += aVal * bVal;
      }
    }

    // #pragma unroll
    // for (int i=0; i<ARowsPerBlock; i++){
    //   int64_t iM = mStart + i;

    //   newPairs[i] = { 0.f, iN };
    //   if (iM < m && iN < n){
    //     #pragma unroll
    //     for (int j=0; j<MaxNNZPerRow; j++){
    //       if (j >= aRowNNZ[i]) continue;

    //       int iK = smemAColInds.get(i, j);
    //       float aVal = smemAVals.get(i, j);
    //       int64_t key[2] = {iN, iK};
    //       int64_t lastVisitedNode = -1;
    //       bool isFound = false;
    //       float bVal = 0.f;
    //       hashmap.find_node(key, lastVisitedNode, isFound);
    //       if (isFound){
    //         bVal = pBVals[lastVisitedNode];
    //       }
    //       newPairs[i].value += aVal * bVal;
    //     }
    //   } else {
    //     newPairs[i].value = -INFINITY;
    //     newPairs[i].index = -3;
    //   }
    //   smemSortTrigger.set(i, 0);
    // }
    __syncthreads();

    // push new (index, value) pair to stack
    pair oldPairs[ARowsPerBlock];
    #pragma unroll
    for (int i=0; i<ARowsPerBlock; i++){
      oldPairs[i] = { -INFINITY, -4 };
      if (threadTopkStack[i].is_full()){
        threadTopkStack[i].pop(oldPairs[i]);
        if (oldPairs[i].value > threadMinValue[i]){
          smemSortTrigger.set(i, 1);
        }
      }

      if (newPairs[i].value > threadMinValue[i]){
        threadTopkStack[i].push(newPairs[i]);
      }
    }
    __syncthreads();

    // sort if necessary
    #pragma unroll
    for (int i=0; i<ARowsPerBlock; i++){
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
  for (int i=0; i<ARowsPerBlock; i++){
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
  for (int i=0; i<ARowsPerBlock; i++){
    int iM = mStart + i;
    if (iM >= m) continue;
    int candIndex = tid;
    pTopkVals[iM * TPB + candIndex] = topkVal[i];
    pTopkInds[iM * TPB + candIndex] = topkInd[i];
  }
}