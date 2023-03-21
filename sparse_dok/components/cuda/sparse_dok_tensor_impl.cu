using value_t = _VALUETYPE_;

extern "C"
__global__ void sparse_dok_count_items(
  const ll_t* pHashmapArgs, //
  const ll_t* pAccessorArgs, //[(1 + 2 * selectorNdim) * nDim]
  const ll_t* pRealSelectorAccessor, //[(1 + 2 * selectorNdim)]
  ll_t* pCounts, //[]
  ll_t nSelectorElements
){
  constexpr int TPB = _TPB_;
  constexpr int ElementsPerThread = _EPT_;
  constexpr int SelectorNDIM = _SELECTORNDIM_;
  constexpr int NDIM = _NDIM_;

  int tid = threadIdx.x;
  int lid = tid % 32;
  int wid = tid / 32;
  ll_t selectorElementOffsetStart = ( blockIdx.x * TPB + tid ) * ElementsPerThread;

  ClosedHashmap<ll_t, value_t, NDIM, 1> hashmap(pHashmapArgs);
  StrictTensorAccessor<ll_t, SelectorNDIM> realSelectorAccessor(pRealSelectorAccessor);
  StrictTensorAccessor<ll_t, SelectorNDIM> selectorAccessors[NDIM];
  
  #pragma unroll
  for (int i=0; i<NDIM; i++){
    selectorAccessors[i].initialize(&pAccessorArgs[ (1 + 2 * SelectorNDIM) * i ]);
  }

  int threadCounts = 0;

  #pragma unroll
  for (int i=0; i<ElementsPerThread; i++){
    ll_t selectorElementOffset = selectorElementOffsetStart + i;
    if (selectorElementOffset < nSelectorElements){
      ll_t selectorElementIdx[SelectorNDIM];
      realSelectorAccessor.get_index_from_offset(selectorElementOffset, selectorElementIdx);
      ll_t index[NDIM];
      #pragma unroll
      for (int j=0; j<NDIM; j++){
        index[j] = selectorAccessors[j].get(selectorElementIdx);
      }

      // value_t value[1];
      // value_t fallbackValue[1] = {0};
      // bool is_found = hashmap.get(index, value, fallbackValue);
      if (hashmap.exists(index)){
        threadCounts++;
      }
    }
  }
  warp_sum<int, 32>(threadCounts);
  if (lid == 0){
    atomicAdd(pCounts, threadCounts);
  }
}

extern "C"
__global__ void sparse_dok_zero_items(
  const ll_t* pHashmapArgs, //
  const ll_t* pAccessorArgs, //[(1 + 2 * selectorNdim) * nDim]
  const ll_t* pRealSelectorAccessor, //[(1 + 2 * selectorNdim)]
  ll_t* pNumRemoved, //[]
  ll_t nSelectorElements
){
  constexpr int TPB = _TPB_;
  constexpr int ElementsPerThread = _EPT_;
  constexpr int SelectorNDIM = _SELECTORNDIM_;
  constexpr int NDIM = _NDIM_;

  int tid = threadIdx.x;
  int lid = tid % 32;
  int wid = tid / 32;
  ll_t selectorElementOffsetStart = ( blockIdx.x * TPB + tid ) * ElementsPerThread;

  ClosedHashmap<ll_t, value_t, NDIM, 1> hashmap(pHashmapArgs);
  StrictTensorAccessor<ll_t, SelectorNDIM> realSelectorAccessor(pRealSelectorAccessor);
  StrictTensorAccessor<ll_t, SelectorNDIM> selectorAccessors[NDIM];
  
  #pragma unroll
  for (int i=0; i<NDIM; i++){
    selectorAccessors[i].initialize(&pAccessorArgs[ (1 + 2 * SelectorNDIM) * i ]);
  }

  int threadCounts = 0;

  #pragma unroll
  for (int i=0; i<ElementsPerThread; i++){
    ll_t selectorElementOffset = selectorElementOffsetStart + i;
    if (selectorElementOffset < nSelectorElements){
      ll_t selectorElementIdx[SelectorNDIM];
      realSelectorAccessor.get_index_from_offset(selectorElementOffset, selectorElementIdx);
      ll_t index[NDIM];
      #pragma unroll
      for (int j=0; j<NDIM; j++){
        index[j] = selectorAccessors[j].get(selectorElementIdx);
      }

      if (hashmap.remove(index)){
        threadCounts++;
      }
    }
  }
  warp_sum<int, 32>(threadCounts);
  if (lid == 0){
    atomicAdd(pNumRemoved, threadCounts);
  }
}

extern "C"
__global__ void sparse_dok_get_items(
  const ll_t* pHashmapArgs, //
  const ll_t* pAccessorArgs, //[(1 + 2 * selectorNdim) * nDim]
  const ll_t* pRealSelectorAccessor, //[(1 + 2 * selectorNdim)]
  const ll_t* pOutHashmapArgs,
  ll_t nSelectorElements
){
  constexpr int TPB = _TPB_;
  constexpr int ElementsPerThread = _EPT_;
  constexpr int SelectorNDIM = _SELECTORNDIM_;
  constexpr int NDIM = _NDIM_;

  int tid = threadIdx.x;
  int lid = tid % 32;
  int wid = tid / 32;
  ll_t selectorElementOffsetStart = ( blockIdx.x * TPB + tid ) * ElementsPerThread;

  ClosedHashmap<ll_t, value_t, NDIM, 1> hashmap(pHashmapArgs);
  ClosedHashmap<ll_t, value_t, SelectorNDIM, 1> outHashmap(pOutHashmapArgs);
  StrictTensorAccessor<ll_t, SelectorNDIM> realSelectorAccessor(pRealSelectorAccessor);
  StrictTensorAccessor<ll_t, SelectorNDIM> selectorAccessors[NDIM];
  
  #pragma unroll
  for (int i=0; i<NDIM; i++){
    selectorAccessors[i].initialize(&pAccessorArgs[ (1 + 2 * SelectorNDIM) * i ]);
  }

  int threadCounts = 0;

  #pragma unroll
  for (int i=0; i<ElementsPerThread; i++){
    ll_t selectorElementOffset = selectorElementOffsetStart + i;
    if (selectorElementOffset < nSelectorElements){
      ll_t selectorElementIdx[SelectorNDIM];
      realSelectorAccessor.get_index_from_offset(selectorElementOffset, selectorElementIdx);
      ll_t index[NDIM];
      #pragma unroll
      for (int j=0; j<NDIM; j++){
        index[j] = selectorAccessors[j].get(selectorElementIdx);
      }

      value_t value[1];
      value_t fallbackValue[1] = {0};
      bool is_found = hashmap.get(index, value, fallbackValue);
      if (is_found){
        outHashmap.set(selectorElementIdx, value);
      }
    }
  }
}

extern "C"
__global__ void sparse_dok_set_items_sparse_v1(
  const ll_t* pSrcIndices, //[selectorNdim, nSrcElements]
  const value_t* pSrcValues, //[nSrcElements]
  const ll_t* pHashmapArgs, //
  const ll_t* pAccessorArgs, //[(1 + 2 * selectorNdim) * nDim]
  const ll_t* pRealSelectorAccessor, //[(1 + 2 * selectorNdim)]
  ll_t nSelectorElements,
  ll_t nSrcElements
){
  constexpr int TPB = _TPB_;
  constexpr int ElementsPerThread = _EPT_;
  constexpr int SelectorNDIM = _SELECTORNDIM_;
  constexpr int NDIM = _NDIM_;

  int tid = threadIdx.x;
  int lid = tid % 32;
  int wid = tid / 32;
  ll_t srcElementOffsetStart = ( blockIdx.x * TPB + tid ) * ElementsPerThread;

  ClosedHashmap<ll_t, value_t, NDIM, 1> destHashmap(pHashmapArgs);
  StrictTensorAccessor<ll_t, SelectorNDIM> realSelectorAccessor(pRealSelectorAccessor);
  StrictTensorAccessor<ll_t, SelectorNDIM> selectorAccessors[NDIM];
  
  #pragma unroll
  for (int i=0; i<NDIM; i++){
    selectorAccessors[i].initialize(&pAccessorArgs[ (1 + 2 * SelectorNDIM) * i ]);
  }

  #pragma unroll
  for (int i=0; i<ElementsPerThread; i++){
    ll_t srcElementOffset = srcElementOffsetStart + i;
    if (srcElementOffset < nSrcElements){
      ll_t selectorElementIndex[SelectorNDIM];
      #pragma unroll
      for (int j=0; j<SelectorNDIM; j++){
        selectorElementIndex[j] = pSrcIndices[nSrcElements * j + srcElementOffset];
      }
      value_t value[1] = { pSrcValues[srcElementOffset] };
      ll_t index[NDIM];
      #pragma unroll
      for (int j=0; j<NDIM; j++){
        index[j] = selectorAccessors[j].get(selectorElementIndex);
      }
      if (value[0] != 0){
        destHashmap.set(index, value);
      }
    }
  }
}

extern "C"
__global__ void sparse_dok_set_items_sparse(
  const ll_t* pSrcHashmapArgs, //
  const ll_t* pDestHashmapArgs, //
  const ll_t* pAccessorArgs, //[(1 + 2 * selectorNdim) * nDim]
  const ll_t* pRealSelectorAccessor, //[(1 + 2 * selectorNdim)]
  ll_t* pNumElements,
  ll_t nSelectorElements
){
  constexpr int TPB = _TPB_;
  constexpr int ElementsPerThread = _EPT_;
  constexpr int SelectorNDIM = _SELECTORNDIM_;
  constexpr int NDIM = _NDIM_;

  int tid = threadIdx.x;
  int lid = tid % 32;
  int wid = tid / 32;
  ll_t selectorElementOffsetStart = ( blockIdx.x * TPB + tid ) * ElementsPerThread;

  ClosedHashmap<ll_t, value_t, SelectorNDIM, 1> srcHashmap(pSrcHashmapArgs);
  ClosedHashmap<ll_t, value_t, NDIM, 1> destHashmap(pDestHashmapArgs);
  StrictTensorAccessor<ll_t, SelectorNDIM> realSelectorAccessor(pRealSelectorAccessor);
  StrictTensorAccessor<ll_t, SelectorNDIM> selectorAccessors[NDIM];

  #pragma unroll
  for (int i=0; i<NDIM; i++){
    selectorAccessors[i].initialize(&pAccessorArgs[ (1 + 2 * SelectorNDIM) * i ]);
  }

  #pragma unroll
  for (int i=0; i<ElementsPerThread; i++){
    ll_t selectorElementOffset = selectorElementOffsetStart + i;
    if (selectorElementOffset < nSelectorElements){
      ll_t selectorElementIdx[SelectorNDIM];
      realSelectorAccessor.get_index_from_offset(selectorElementOffset, selectorElementIdx);
      ll_t index[NDIM];
      #pragma unroll
      for (int j=0; j<NDIM; j++){
        index[j] = selectorAccessors[j].get(selectorElementIdx);
      }
      // value_t value[1] = { srcAccessor.get(selectorElementIdx) };
      value_t srcValue[1];
      value_t destValue[1];
      value_t fallbackValue[1] = {0};
      srcHashmap.get(selectorElementIdx, srcValue, fallbackValue);
      
      // destHashmap.get(index, destValue, fallbackValue);
      // if (srcValue[0] == 0 && destValue[0] != 0){
      //   destHashmap.remove(index);
      // } else if (srcValue[0] != 0){
      //   destHashmap.set(index, srcValue);
      // }
      destHashmap.set(index, srcValue);
    }
  }
}

extern "C"
__global__ void sparse_dok_set_items_dense(
  const ll_t* pHashmapArgs, //
  const ll_t* pAccessorArgs, //[(1 + 2 * selectorNdim) * nDim]
  const ll_t* pRealSelectorAccessor, //[(1 + 2 * selectorNdim)]
  ll_t nSelectorElements
){
  constexpr int TPB = _TPB_;
  constexpr int ElementsPerThread = _EPT_;
  constexpr int SelectorNDIM = _SELECTORNDIM_;
  constexpr int NDIM = _NDIM_;

  int tid = threadIdx.x;
  int lid = tid % 32;
  int wid = tid / 32;
  ll_t selectorElementOffsetStart = ( blockIdx.x * TPB + tid ) * ElementsPerThread;

  ClosedHashmap<ll_t, value_t, NDIM, 1> destHashmap(pHashmapArgs);
  StrictTensorAccessor<ll_t, SelectorNDIM> realSelectorAccessor(pRealSelectorAccessor);
  StrictTensorAccessor<ll_t, SelectorNDIM> selectorAccessors[NDIM];
  StrictTensorAccessor<value_t, SelectorNDIM> srcAccessor(&pAccessorArgs[(1 + 2 * SelectorNDIM) * NDIM]);

  
  #pragma unroll
  for (int i=0; i<NDIM; i++){
    selectorAccessors[i].initialize(&pAccessorArgs[ (1 + 2 * SelectorNDIM) * i ]);
  }

  #pragma unroll
  for (int i=0; i<ElementsPerThread; i++){
    ll_t selectorElementOffset = selectorElementOffsetStart + i;
    if (selectorElementOffset < nSelectorElements){
      ll_t selectorElementIdx[SelectorNDIM];
      realSelectorAccessor.get_index_from_offset(selectorElementOffset, selectorElementIdx);
      ll_t index[NDIM];
      #pragma unroll
      for (int j=0; j<NDIM; j++){
        index[j] = selectorAccessors[j].get(selectorElementIdx);
      }
      value_t value[1] = { srcAccessor.get(selectorElementIdx) };
      destHashmap.set(index, value);
    }
  }
}