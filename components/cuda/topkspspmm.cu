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
  constexpr int ARPB = _ARPB_; // A rows per block
  constexpr int MaxNNZPerRow = _MAXNNZPR_; // max number of nonzeros per row
  
  int tid = threadIdx.x;
  const int mStart = blockIdx.x * ARPB;

  // load rows inds and values from A
  extern __shared__ int64_t smemPtr[];
  SmemTensor2D<int64_t, ARPB, MaxNNZPerRow> smemAColInds(reinterpret_cast<int64_t*>(smemPtr));       //[ARPB, maxnnzpr]
  SmemTensor2D<float, ARPB, MaxNNZPerRow> smemAVals(reinterpret_cast<float*>(smemAColInds.endPtr));  //[ARPB, maxnnzpr]
  
  SmemTensor1D<float, TPB> smemValueExchange(reinterpret_cast<float*>(smemAVals.endPtr));  //[TPB]
  SmemTensor1D<int64_t, TPB> smemIndexExchange(reinterpret_cast<int64_t*>(smemValueExchange.endPtr));  //[TPB]

  int aRowNNZ[ARPB] = {0};
  #pragma unroll
  for (int i=0; i<ARPB; i++){
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
  __syncthreads();

  // initialize hashmap
  Hashmap<2, int64_t> hashmap(pAlpha1, pAlpha2,
                              pBeta1, pBeta2,
                              pPrime1, pPrime2,
                              pHeadNode, pNextNode,
                              pUUID, nBuckets);
  
  // start iterating over rows of B
  // each thread is responsible for one row
  // no need to worry about uncoalesced memory access, because it's random access anyways. 
  float topkVal[ARPB] = {-INFINITY};
  int64_t topkInd[ARPB] = {-1};
  for (int itr=0; itr < div_ru(n, TPB); itr ++){
    int64_t iN = itr * TPB + tid;
    // if (iN >= n) continue;

    #pragma unroll
    for (int i=0; i<ARPB; i++){
      int64_t iM = mStart + i;
      // if (iM >= m) continue;

      float tempVal = 0.f;
      int64_t tempInd = iN;
      if (iM < m && iN < n){
        #pragma unroll
        for (int j=0; j<MaxNNZPerRow; j++){
          if (j >= aRowNNZ[i]) continue;

          int iK = smemAColInds.get(i, j);
          float aVal = smemAVals.get(i, j);
          int64_t key[2] = {iN, iK};
          int64_t lastVisitedNode = -1;
          bool isFound = false;
          float bVal = 0.f;
          hashmap.find_node(key, lastVisitedNode, isFound);
          if (isFound){
            bVal = pBVals[lastVisitedNode];
          }
          tempVal += aVal * bVal;
        }
      } else {
        tempVal = -INFINITY;
        tempInd = -1;
      }
      // TODO:
      bitonic_sort<TPB>(tempVal, tempInd, 
                        smemValueExchange.startPtr, smemIndexExchange.startPtr, 
                        tid);
      // topkVal[i] = tempVal;
      // topkInd[i] = tempInd;
      // if (tempVal == 0.f){
      //   tempInd = 5;
      // }
      bitonic_sort_global<TPB>(topkVal[i], topkInd[i], 
                              tempVal, tempInd, 
                              smemValueExchange.startPtr, smemIndexExchange.startPtr,
                              tid);

    }

    // write results back
  }
  #pragma unroll
  for (int i=0; i<ARPB; i++){
    int64_t iM = mStart + i;
    if (iM >= m) continue;
    int candIndex = tid;
    pTopkInds[iM * TPB + candIndex] = topkInd[i];
    pTopkVals[iM * TPB + candIndex] = topkVal[i];
  }
}