
using KeyType = _KEYTYPE_;
using ValueType = _VALUETYPE_;
using BoolType = uint8_t;

extern "C"
__global__ void closed_hashmap_get(
  const ll_t* __restrict__ pPrime1, //[KeySize]
  const ll_t* __restrict__ pPrime2, //[KeySize]
  const ll_t* __restrict__ pAlpha1, //[KeySize]
  const ll_t* __restrict__ pAlpha2, //[KeySize]
  const ll_t* __restrict__ pBeta1,  //[KeySize]
  const ll_t* __restrict__ pBeta2,  //[KeySize]
  const ll_t* __restrict__ pKeyPerm,             //[KeySize]
  const KeyType* __restrict__ pKeys,             //[NumKeys, KeySize]
  ValueType* pValues,         //[NumKeys, ValueSize]
  KeyType* pAllKeys,          //[NumBuckets, KeySize]
  ValueType* pAllValues,      //[NumBuckets, ValueSize]
  ll_t* pAllUUIDs,            //[NumBuckets]
  const ValueType* __restrict__ pFallbackValue,  //[ValueSize]  
  BoolType* pIsFound,        //[NumKeys]
  ll_t numKeys, ll_t numBuckets
){
  constexpr int TPB = _TPB_;
  constexpr int KPT = _KPT_;
  constexpr int KeySize = _KEYSIZE_;
  constexpr int ValueSize = _VALUESIZE_;
  constexpr int KPB = TPB * KPT;

  int tid = threadIdx.x;
  ll_t kStart = blockIdx.x * KPB;

  ClosedHashmap<KeyType, ValueType, KeySize, ValueSize> hashmap(
    pPrime1, pPrime2,
    pAlpha1, pAlpha2,
    pBeta1,  pBeta2,
    pKeyPerm,
    pAllKeys,
    pAllValues,
    pAllUUIDs,
    numBuckets,
    -1,
    -3
  );

  // Load keys
  KeyType keys[KPT][KeySize];
  ValueType values[KPT][ValueSize];
  ValueType fallbackValue[ValueSize];
  #pragma unroll
  for (int i=0; i<KPT; i++){
    ll_t offset = kStart + i * TPB + tid;
    if (offset < numKeys){
      #pragma unroll
      for (int j=0; j<KeySize; j++){
        keys[i][j] = pKeys[offset * KeySize + j];
        // keys[i][j] = pKeys[offset * KeySize + hashmap.keyPerm[j]];
      }
    }
  }
  
  #pragma unroll
  for (int i=0; i<ValueSize; i++){
    fallbackValue[i] = pFallbackValue[i];
  }

  // get values
  bool isFound[KPT];
  // hashmap.get_batched<KPT>(keys, values, fallbackValue, isFound);
  #pragma unroll
  for (int i=0; i<KPT; i++){
    int offset = kStart + i * TPB + tid;
    if (offset < numKeys){
      isFound[i] = hashmap.get(keys[i], values[i], fallbackValue);

      pIsFound[offset] = (BoolType) isFound[i];
      if (isFound[i]){
        #pragma unroll
        for (int j=0; j<ValueSize; j++){
          pValues[offset * ValueSize + j] = values[i][j];
        }
      }
    }
  }
}

extern "C"
__global__ void closed_hashmap_set(
  const ll_t* __restrict__ pPrime1, //[KeySize]
  const ll_t* __restrict__ pPrime2, //[KeySize]
  const ll_t* __restrict__ pAlpha1, //[KeySize]
  const ll_t* __restrict__ pAlpha2, //[KeySize]
  const ll_t* __restrict__ pBeta1,  //[KeySize]
  const ll_t* __restrict__ pBeta2,  //[KeySize]
  const ll_t* __restrict__ pKeyPerm,             //[KeySize]
  const KeyType* __restrict__ pKeys,             //[NumKeys, KeySize]
  const ValueType* __restrict__ pValues,         //[NumKeys, ValueSize]
  KeyType* pAllKeys,          //[NumBuckets, KeySize]
  ValueType* pAllValues,      //[NumBuckets, ValueSize]
  ll_t* pAllUUIDs,            //[NumBuckets]
  BoolType* pIsStored,        //[NumKeys]
  ll_t numKeys, ll_t numBuckets
){
  constexpr int TPB = _TPB_;
  constexpr int KPT = _KPT_;
  constexpr int KeySize = _KEYSIZE_;
  constexpr int ValueSize = _VALUESIZE_;
  constexpr int KPB = TPB * KPT;

  int tid = threadIdx.x;
  ll_t kStart = blockIdx.x * KPB;

  ClosedHashmap<KeyType, ValueType, KeySize, ValueSize> hashmap(
    pPrime1, pPrime2,
    pAlpha1, pAlpha2,
    pBeta1,  pBeta2,
    pKeyPerm,
    pAllKeys,
    pAllValues,
    pAllUUIDs,
    numBuckets,
    -1,
    -3
  );

  // Load keys
  KeyType keys[KPT][KeySize];
  ValueType values[KPT][ValueSize];
  #pragma unroll
  for (int i=0; i<KPT; i++){
    ll_t offset = kStart + i * TPB + tid;
    if (offset < numKeys){
      #pragma unroll
      for (int j=0; j<KeySize; j++){
        keys[i][j] = pKeys[offset * KeySize + j];      
      }
      #pragma unroll
      for (int j=0; j<ValueSize; j++){
        values[i][j] = pValues[offset * ValueSize + j];
      }
    }
  }

  // get values
  bool isStored[KPT];
  // hashmap.set_batched<KPT>(keys, values, isStored);

  #pragma unroll
  for (int i=0; i<KPT; i++){
    int offset = kStart + i * TPB + tid;
    if (offset < numKeys){
      isStored[i] = hashmap.set(keys[i], values[i]);
      pIsStored[offset] = (BoolType) isStored[i];
    }
  }
}

extern "C"
__global__ void closed_hashmap_remove(
  const ll_t* __restrict__ pPrime1, //[KeySize]
  const ll_t* __restrict__ pPrime2, //[KeySize]
  const ll_t* __restrict__ pAlpha1, //[KeySize]
  const ll_t* __restrict__ pAlpha2, //[KeySize]
  const ll_t* __restrict__ pBeta1,  //[KeySize]
  const ll_t* __restrict__ pBeta2,  //[KeySize]
  const ll_t* __restrict__ pKeyPerm,             //[KeySize]
  const KeyType* __restrict__ pKeys,             //[NumKeys, KeySize]
  KeyType* pAllKeys,          //[NumBuckets, KeySize]
  ValueType* pAllValues,      //[NumBuckets, ValueSize]
  ll_t* pAllUUIDs,            //[NumBuckets]
  BoolType* pIsRemoved,        //[NumKeys]
  ll_t numKeys, ll_t numBuckets
){
  constexpr int TPB = _TPB_;
  constexpr int KPT = _KPT_;
  constexpr int KeySize = _KEYSIZE_;
  constexpr int ValueSize = _VALUESIZE_;
  constexpr int KPB = TPB * KPT;

  int tid = threadIdx.x;
  ll_t kStart = blockIdx.x * KPB;

  ClosedHashmap<KeyType, ValueType, KeySize, ValueSize> hashmap(
    pPrime1, pPrime2,
    pAlpha1, pAlpha2,
    pBeta1,  pBeta2,
    pKeyPerm,
    pAllKeys,
    pAllValues,
    pAllUUIDs,
    numBuckets,
    -1,
    -3
  );

  // Load keys
  KeyType keys[KPT][KeySize];
  #pragma unroll
  for (int i=0; i<KPT; i++){
    ll_t offset = kStart + i * TPB + tid;
    if (offset < numKeys){
      #pragma unroll
      for (int j=0; j<KeySize; j++){
        keys[i][j] = pKeys[offset * KeySize + j];
        // keys[i][j] = pKeys[offset * KeySize + hashmap.keyPerm[j]];
      }
    }
  }
  
  // remove
  bool isRemoved[KPT];
  #pragma unroll
  for (int i=0; i<KPT; i++){
    int offset = kStart + i * TPB + tid;
    if (offset < numKeys){
      isRemoved[i] = hashmap.remove(keys[i]);
      pIsRemoved[offset] = (BoolType) isRemoved[i];
    }
  }
}