#define EMPTY 1
#define FOUND 2
#define NOT_FOUND 3
#define STORED 4
#define NOT_STORED 5

template <
  typename KeyType,
  typename ValueType,
  int KeySize,
  int ValueSize
>
class ClosedHashmap{
  private:
    ll_t _prime1[KeySize];
    ll_t _prime2[KeySize];
    ll_t _alpha1[KeySize];
    ll_t _alpha2[KeySize];
    ll_t _beta1[KeySize];
    ll_t _beta2[KeySize];
    KeyType *_pAllKeys;
    ValueType *_pAllValues;
    ll_t *_pAllUUIDs;
    ll_t _numBuckets;
    ll_t _emptyMarker;
    ll_t _removedMarker;

  public:
    ll_t keyPerm[KeySize];

    CUDA_DEVICE_INLINE
    ClosedHashmap(const ll_t* pPrime1,
                  const ll_t* pPrime2,
                  const ll_t* pAlpha1,
                  const ll_t* pAlpha2,
                  const ll_t* pBeta1,
                  const ll_t* pBeta2,
                  const ll_t* pKeyPerm,
                  KeyType* pAllKeys,
                  ValueType* pAllValues,
                  ll_t* pAllUUIDs,
                  ll_t numBuckets,
                  ll_t emptyMarker,
                  ll_t removedMarker
                  )
                  : _pAllKeys(pAllKeys)
                  , _pAllValues(pAllValues)
                  , _pAllUUIDs(pAllUUIDs)
                  , _numBuckets(numBuckets)
                  , _emptyMarker(emptyMarker)
                  , _removedMarker(removedMarker)
    {
      #pragma unroll
      for (int i=0; i < KeySize; i++){
        keyPerm[i] = pKeyPerm[i];
        _prime1[i] = pPrime1[keyPerm[i]];
        _prime2[i] = pPrime2[keyPerm[i]];
        _alpha1[i] = pAlpha1[keyPerm[i]];
        _alpha2[i] = pAlpha2[keyPerm[i]];
        _beta1[i] = pBeta1[keyPerm[i]];
        _beta2[i] = pBeta2[keyPerm[i]];
      }
    }

    CUDA_DEVICE_INLINE
    ll_t get_hash(KeyType key[KeySize]){
      ll_t hash_code = ( (ll_t) key[0] * _alpha1[0] + _beta1[0]) % _prime1[0];
      #pragma unroll
      for (int i=1; i<KeySize; i++){
        hash_code *= ( (ll_t) key[i] * _alpha1[i] + _beta1[i]) % _prime1[i];
      }
      hash_code = llabs(hash_code);
      return hash_code;
    }

    CUDA_DEVICE_INLINE
    ll_t get_uuid(KeyType key[KeySize]){
      ll_t uuid = ( (ll_t) key[0] * _alpha2[0] + _beta2[0]) % _prime2[0];
      #pragma unroll
      for (int i=1; i<KeySize; i++){
        uuid *= ( (ll_t) key[i] * _alpha2[i] + _beta2[i]) % _prime2[i];
      }
      uuid = llabs(uuid);
      return uuid;
    }

    CUDA_DEVICE_INLINE
    bool are_keys_equal(KeyType key1[KeySize], KeyType key2[KeySize]){
      bool isEqual = key1[0] == key2[0];
      #pragma unroll
      for (int i=0; i<KeySize; i++){
        isEqual = isEqual && (key1[i] == key2[i]);
      }
      return isEqual;
    }

    CUDA_DEVICE_INLINE
    void get_key(ll_t address, KeyType key[KeySize]){
      #pragma unroll
      for (int i=0; i<KeySize; i++){
        key[i] = _pAllKeys[address * KeySize + i];
      }
    }

    CUDA_DEVICE_INLINE
    void get_key_permuted(ll_t address, KeyType key[KeySize]){
      #pragma unroll
      for (int i=0; i<KeySize; i++){
        key[i] = _pAllKeys[address * KeySize + keyPerm[i]];
      }
    }

    CUDA_DEVICE_INLINE
    void set_key(ll_t address, KeyType key[KeySize]){
      #pragma unroll
      for (int i=0; i<KeySize; i++){
        _pAllKeys[address * KeySize + i] = key[i];
      }
    
    }

    CUDA_DEVICE_INLINE
    void set_key_permuted(ll_t address, KeyType key[KeySize]){
      #pragma unroll
      for (int i=0; i<KeySize; i++){
        _pAllKeys[address * KeySize + keyPerm[i] ] = key[i];
      }
    }

    CUDA_DEVICE_INLINE
    void get_value(ll_t address, ValueType value[ValueSize]){
      #pragma unroll
      for (int i=0; i<ValueSize; i++){
        value[i] = _pAllValues[address * ValueSize + i];
      }
    }

    CUDA_DEVICE_INLINE
    void set_value(ll_t address, ValueType value[ValueSize]){
      #pragma unroll
      for (int i=0; i<ValueSize; i++){
        _pAllValues[address * ValueSize + i] = value[i];
      }
    }

    CUDA_DEVICE_INLINE
    void get_uuid(ll_t address, ll_t &uuid){
      uuid = _pAllUUIDs[address];
    }

    CUDA_DEVICE_INLINE
    ll_t get_uuid(ll_t address){
      return _pAllUUIDs[address];
    }

    CUDA_DEVICE_INLINE
    void set_uuid(ll_t address, ll_t uuid){
      _pAllUUIDs[address] = uuid;
    }

    CUDA_DEVICE_INLINE
    bool set_uuid_if_empty(int address, ll_t uuid, ll_t &oldUUID){
      ll_t *ptr = &_pAllUUIDs[address];
      // if the value at `ptr` is equal to `_emptyMarker`, then set the value of that pointer to `uuid`, return true
      // else, return false
      oldUUID = atomicCAS(ptr, _emptyMarker, uuid);
      if ( oldUUID != _emptyMarker){
        return false;
      }
      return true;
    }

    CUDA_DEVICE_INLINE
    bool set_uuid_if_removed(int address, ll_t uuid, ll_t &oldUUID){
      ll_t *ptr = &_pAllUUIDs[address];
      // if the value at `ptr` is equal to `_removedMarker`, then set the value of that pointer to `uuid`, return true
      // else, return false
      oldUUID = atomicCAS(ptr, _removedMarker, uuid);
      if ( oldUUID != _removedMarker){
        return false;
      }
      return true;
    }

    CUDA_DEVICE_INLINE
    int get_by_uuid(ll_t address, ll_t uuid, ValueType value[ValueSize]){
      ll_t candidateUUID = get_uuid(address);
      // check if the candidateKey is emptyKey
      bool isEmpty = candidateUUID == _emptyMarker;
      // is so, return not found
      if (isEmpty){
        return EMPTY;
      }
      // check if the candidateKey is equal to key
      bool isFound = candidateUUID == uuid;
      // if so, return found
      if (isFound){
        get_value(address, value);
        return FOUND;
      }
      return NOT_FOUND;
    }

     CUDA_DEVICE_INLINE
    int set_by_uuid(int address, ll_t uuid, KeyType key[KeySize], ValueType value[ValueSize]){
      // is so, store key and value in this address
      // set key to that address, if storing failed (because of another thread using that address ), return not stored
      ll_t candidateUUID;
      bool isSuccessful = set_uuid_if_empty(address, uuid, candidateUUID);
      if (isSuccessful){
        set_key(address, key);
        set_value(address, value);
        return STORED;
      }
      // check if the candidateUUID is equal to uuid
      bool isFound = uuid == candidateUUID;
      // if so, return stored
      if (isFound){
        set_key(address, key);
        set_value(address, value);
        return STORED;
      }
      // otherwise, return not found
      return NOT_STORED;
    }

    CUDA_DEVICE_INLINE
    bool get(
      KeyType key[KeySize],
      ValueType value[ValueSize],
      ValueType fallbackValue[ValueSize]
    ){
      // permute_key(key);
      ll_t hashCode = get_hash(key);
      ll_t uuid = get_uuid(key);
      #pragma unroll 2
      for (ll_t i=0; i < _numBuckets; i++){
        ll_t address = (hashCode + i) % _numBuckets;
        ll_t candidateUUID = get_uuid(address);
        // check if the candidateKey is emptyKey
        bool isEmpty = candidateUUID == _emptyMarker;
        // is so, return not found
        if (isEmpty){
          break;
        }
        // check if the candidateKey is equal to key
        bool isFound = candidateUUID == uuid;
        // if so, return found
        if (isFound){
          get_value(address, value);
          return true;
        }
      }
      #pragma unroll
      for (int j=0; j<ValueSize; j++){
        value[j] = fallbackValue[j];
      }
      return false;
    }

    CUDA_DEVICE_INLINE
    bool set(
      KeyType key[KeySize],
      ValueType value[ValueSize]
    ){
      // permute_key(key);
      ll_t hashCode = get_hash(key);
      ll_t uuid = get_uuid(key);
      ll_t firstRemovedAddress = -1;
      #pragma unroll 2
      for (ll_t i=0; i<_numBuckets; i++){
        ll_t address = (hashCode + i) % _numBuckets;
        ll_t candidateUUID;
        candidateUUID = get_uuid(address);
        bool isFound = candidateUUID == uuid;
        // if key is found, return stored
        if (isFound){
          set_key_permuted(address, key);
          set_value(address, value);
          return true;
        }

        bool isRemoved = candidateUUID == _removedMarker;
        if (isRemoved && firstRemovedAddress == -1){
          firstRemovedAddress = address;
        }

        bool isEmpty = candidateUUID == _emptyMarker;
        if (isEmpty){
          // if no deletedMarker encountered previously, store key-value pair to nearest empty address.
          if (firstRemovedAddress == -1){
            bool isSuccessful = set_uuid_if_empty(address, uuid, candidateUUID);
            if (isSuccessful){
              set_key_permuted(address, key);
              set_value(address, value);
              return true;
            }
          } else {
          // otherwise, try to store the key-value pair to that deletedMarker, if fail, store to nearest empty address.
            bool isSuccessful = set_uuid_if_removed(firstRemovedAddress, uuid, candidateUUID);
            if (isSuccessful){
              set_key_permuted(firstRemovedAddress, key);
              set_value(firstRemovedAddress, value);
              return true;
            } else {
              firstRemovedAddress = -1;
              i--;
            }
          }
        }
      }
      return false;
    }

    CUDA_DEVICE_INLINE
    bool set_old(
      KeyType key[KeySize],
      ValueType value[ValueSize]
    ){
      // permute_key(key);
      ll_t hashCode = get_hash(key);
      ll_t uuid = get_uuid(key);
      #pragma unroll 2
      for (ll_t i=0; i<_numBuckets; i++){
        ll_t address = (hashCode + i) % _numBuckets;
        ll_t candidateUUID;
        bool isSuccessful = set_uuid_if_empty(address, uuid, candidateUUID);
        if (isSuccessful){
          set_key_permuted(address, key);
          set_value(address, value);
          return true;
        }
        // check if the candidateUUID is equal to uuid
        bool isFound = uuid == candidateUUID;
        // if so, return stored
        if (isFound){
          set_key_permuted(address, key);
          set_value(address, value);
          return true;
        }
      }
      return false;
    }

    CUDA_DEVICE_INLINE
    bool remove(
      KeyType key[KeySize]
    ){
      ll_t hashCode = get_hash(key);
      ll_t uuid = get_uuid(key);
      #pragma unroll 2
      for (ll_t i=0; i < _numBuckets; i++){
        ll_t address = (hashCode + i) % _numBuckets;
        ll_t candidateUUID = get_uuid(address);
        // check if the candidateKey is emptyKey
        bool isEmpty = candidateUUID == _emptyMarker;
        // is so, return not found
        if (isEmpty){
          break;
        }
        // check if the candidateKey is equal to key
        bool isFound = candidateUUID == uuid;
        // if so, return found
        if (isFound){
          set_uuid(address, _removedMarker);
          return true;
        }
      }
      return false;
    }

    template <int BatchSize>
    CUDA_DEVICE_INLINE
    void get_batched(
      KeyType key[BatchSize][KeySize],
      ValueType value[BatchSize][ValueSize],
      ValueType fallbackValue[ValueSize],
      bool isFound[BatchSize]
    ){
      ll_t hashCode[BatchSize];
      ll_t uuid[BatchSize];
      bool isDone[BatchSize];
      ll_t address[BatchSize];
      #pragma unroll
      for (int b = 0; b < BatchSize; b++){
        hashCode[b] = get_hash(key[b]);
        uuid[b] = get_uuid(key[b]);
        isDone[b] = false;
        isFound[b] = false;
      }
      #pragma unroll 2
      for (ll_t i=0; i < _numBuckets; i++){
        ll_t candidateUUID[BatchSize];
        #pragma unroll
        for (int b = 0; b < BatchSize; b++){
          address[b] = (hashCode[b] + i) % _numBuckets;
          candidateUUID[b] = get_uuid(address[b]);
        }
        #pragma unroll
        for (int b = 0; b < BatchSize; b++){
          // check if the candidateKey is emptyKey
          bool isEmpty = candidateUUID[b] == _emptyMarker;
          // is so, return not found
          if (isEmpty){
            isDone[b] = true;
          }
          // check if the candidateKey is equal to key
          isFound[b] = candidateUUID[b] == uuid[b];
          // if so, return found
          if (isFound[b]){
            get_value(address[b], value[b]);
            // return true;
            isDone[b] = true;
          }
        }
        bool isAllDone = isDone[0];
        #pragma unroll
        for (int b=1; b < BatchSize; b++){
          isAllDone = isAllDone && isDone[b];
        }
        if (isAllDone){
          break;
        }
      }
      #pragma unroll
      for (int b=0; b<BatchSize; b++){
        if (!isFound[b]){
          #pragma unroll
          for (int j=0; j<ValueSize; j++){
            value[b][j] = fallbackValue[j];
          }
        }
      }
    }

    template <int BatchSize>
    CUDA_DEVICE_INLINE
    void set_batched(
      KeyType key[BatchSize][KeySize],
      ValueType value[BatchSize][ValueSize],
      bool isStored[BatchSize]
    ){
      ll_t hashCode[BatchSize];
      ll_t uuid[BatchSize];
      bool isDone[BatchSize];
      #pragma unroll
      for (int b=0; b<BatchSize; b++){
        hashCode[b] = get_hash(key[b]);
        uuid[b] = get_uuid(key[b]);
        isDone[b] = false;
        isStored[b] = false;
      }

      #pragma unroll 2
      for (ll_t i=0; i<_numBuckets; i++){
        ll_t address[BatchSize];
        ll_t candidateUUID[BatchSize];
        bool isSuccessful[BatchSize];
        #pragma unroll
        for (int b=0; b<BatchSize; b++){
          address[b] = (hashCode[b] + i) % _numBuckets;
          isSuccessful[b] = set_uuid_if_empty(address[b], uuid[b], candidateUUID[b]);
        }

        #pragma unroll
        for (int b=0; b<BatchSize; b++){
          isStored[b] = isSuccessful[b] || (uuid[b] == candidateUUID[b]);
          if (isStored[b]){
            set_key(address[b], key[b]);
            set_value(address[b], value[b]);
            isDone[b] = true;
          }
        }

        bool isAllDone = isDone[0];
        #pragma unroll
        for (int b=1; b<BatchSize; b++){
          isAllDone = isAllDone && isDone[b];
        }
        if (isAllDone){
          break;
        }
      }
    }

    // CUDA_DEVICE_INLINE
    // void permute_key(KeyType key[KeySize]){
    //   KeyType permutedKey[KeySize];
    //   #pragma unroll
    //   for (int i=0; i<KeySize; i++){
    //     permutedKey[i] = key[keyPerm[i]];
    //   }
    //   #pragma unroll
    //   for (int i=0; i<KeySize; i++){
    //     key[i] = permutedKey[i];
    //   }
    // }
};


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

// extern "C"
// __global__ void closed_hashmap_count_existing(
//   const ll_t* __restrict__ pPrime1, //[KeySize]
//   const ll_t* __restrict__ pPrime2, //[KeySize]
//   const ll_t* __restrict__ pAlpha1, //[KeySize]
//   const ll_t* __restrict__ pAlpha2, //[KeySize]
//   const ll_t* __restrict__ pBeta1,  //[KeySize]
//   const ll_t* __restrict__ pBeta2,  //[KeySize]
//   const KeyType* __restrict__ pKeys,             //[NumKeys, KeySize]
//   KeyType* pAllKeys,          //[NumBuckets, KeySize]
//   ValueType* pAllValues,      //[NumBuckets, ValueSize]
//   ll_t* pAllUUIDs,            //[NumBuckets]
//   ull_t* __restrict__ pCounts, //[1]
//   ll_t numKeys, ll_t numBuckets
// ){
//   constexpr int TPB = _TPB_;
//   constexpr int KPT = _KPT_;
//   constexpr int KeySize = _KEYSIZE_;
//   constexpr int ValueSize = _VALUESIZE_;
//   constexpr int KPB = TPB * KPT;

//   int tid = threadIdx.x;
//   ll_t kStart = blockIdx.x * KPB;

//   ClosedHashmap<KeyType, ValueType, KeySize, ValueSize> hashmap(
//     pPrime1, pPrime2,
//     pAlpha1, pAlpha2,
//     pBeta1,  pBeta2,
//     pAllKeys,
//     pAllValues,
//     pAllUUIDs,
//     numBuckets,
//     -1
//   );

//   // Load keys
//   KeyType keys[KPT][KeySize];
//   ValueType values[KPT][ValueSize];
//   ValueType fallbackValue[ValueSize];
//   #pragma unroll
//   for (int i=0; i<KPT; i++){
//     ll_t offset = kStart + i * TPB + tid;
//     if (offset < numKeys){
//       #pragma unroll
//       for (int j=0; j<KeySize; j++){
//         keys[i][j] = pKeys[offset * KeySize + j];
//       }
//     }
//   }
  
//   // #pragma unroll
//   // for (int i=0; i<ValueSize; i++){
//   //   fallbackValue[i] = pFallbackValue[i];
//   // }
//   __shared__ int blockCount[1];
//   // get values
//   int threadCount = 0;
//   // bool isFound[KPT];
//   #pragma unroll
//   for (int i=0; i<KPT; i++){
//     int offset = kStart + i * TPB + tid;
//     if (offset < numKeys){
//       bool isFound = hashmap.get(keys[i], values[i], fallbackValue);
//       if (isFound){
//         threadCount ++;
//       }
//     }
//   }

//   atomicAdd(blockCount, threadCount);
//   if (tid == 0){
//     atomicAdd(pCounts, (ull_t) blockCount[0]);
//   }
// }

// extern "C"
// __global__ void closed_hashmap_get_sparse(
//   const ll_t* __restrict__ pPrime1, //[KeySize]
//   const ll_t* __restrict__ pPrime2, //[KeySize]
//   const ll_t* __restrict__ pAlpha1, //[KeySize]
//   const ll_t* __restrict__ pAlpha2, //[KeySize]
//   const ll_t* __restrict__ pBeta1,  //[KeySize]
//   const ll_t* __restrict__ pBeta2,  //[KeySize]
//   const KeyType* __restrict__ pKeys,             //[NumKeys, KeySize]
//   KeyType* pAllKeys,          //[NumBuckets, KeySize]
//   ValueType* pAllValues,      //[NumBuckets, ValueSize]
//   ll_t* pAllUUIDs,            //[NumBuckets] 
  
//   const ll_t* __restrict__ pOutPrime1, //[KeySize]
//   const ll_t* __restrict__ pOutPrime2, //[KeySize]
//   const ll_t* __restrict__ pOutAlpha1, //[KeySize]
//   const ll_t* __restrict__ pOutAlpha2, //[KeySize]
//   const ll_t* __restrict__ pOutBeta1,  //[KeySize]
//   const ll_t* __restrict__ pOutBeta2,  //[KeySize]
//   KeyType* pOutAllKeys,          //[NumBuckets, KeySize]
//   ValueType* pOutAllValues,      //[NumBuckets, ValueSize]
//   ll_t* pOutAllUUIDs,            //[NumBuckets] 

//   ll_t numKeys, ll_t numBuckets, ll_t numOutBuckets
// ){
//   constexpr int TPB = _TPB_;
//   constexpr int KPT = _KPT_;
//   constexpr int KeySize = _KEYSIZE_;
//   constexpr int ValueSize = _VALUESIZE_;
//   constexpr int KPB = TPB * KPT;

//   int tid = threadIdx.x;
//   ll_t kStart = blockIdx.x * KPB;

//   ClosedHashmap<KeyType, ValueType, KeySize, ValueSize> hashmap(
//     pPrime1, pPrime2,
//     pAlpha1, pAlpha2,
//     pBeta1,  pBeta2,
//     pAllKeys,
//     pAllValues,
//     pAllUUIDs,
//     numBuckets,
//     -1
//   );

//   ClosedHashmap<KeyType, ValueType, KeySize, ValueSize> outHashmap(
//     pOutPrime1, pOutPrime2,
//     pOutAlpha1, pOutAlpha2,
//     pOutBeta1,  pOutBeta2,
//     pOutAllKeys,
//     pOutAllValues,
//     pOutAllUUIDs,
//     numOutBuckets,
//     -1
//   );


//   // Load keys
//   KeyType keys[KPT][KeySize];
//   ValueType values[KPT][ValueSize];
//   ValueType fallbackValue[ValueSize];
//   #pragma unroll
//   for (int i=0; i<KPT; i++){
//     ll_t offset = kStart + i * TPB + tid;
//     if (offset < numKeys){
//       #pragma unroll
//       for (int j=0; j<KeySize; j++){
//         keys[i][j] = pKeys[offset * KeySize + j];
//       }
//     }
//   }
  
//   #pragma unroll
//   for (int i=0; i<ValueSize; i++){
//     fallbackValue[i] = pFallbackValue[i];
//   }

//   // get values
//   bool isFound[KPT];
//   // hashmap.get_batched<KPT>(keys, values, fallbackValue, isFound);
//   #pragma unroll
//   for (int i=0; i<KPT; i++){
//     int offset = kStart + i * TPB + tid;
//     if (offset < numKeys){
//       isFound[i] = hashmap.get(keys[i], values[i], fallbackValue);
//       if (isFound[i]){
//         outHashmap.set(keys[i], values[i]);
//       //   #pragma unroll
//       //   for (int j=0; j<ValueSize; j++){
//       //     pValues[offset * ValueSize + j] = values[i][j];
//       //   }
//       }
//     }
//   }
//   #pragma unroll
//   for (int i=0; i<KPT; i++){
//     int offset = kStart + i * TPB + tid;
//     if (offset < numKeys){
// }