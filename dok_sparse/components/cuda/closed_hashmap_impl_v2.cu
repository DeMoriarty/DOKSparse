#define EMPTY 1
#define FOUND 2
#define NOT_FOUND 3
#define STORED 4
#define NOT_STORED 5

template <
  typename KeyType,
  typename ValueType,
  int KeySize,
  int ValueSize,
  int TPB,
  int TPG
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

  public:
    CUDA_DEVICE_INLINE
    ClosedHashmap(const ll_t* pPrime1,
                  const ll_t* pPrime2,
                  const ll_t* pAlpha1,
                  const ll_t* pAlpha2,
                  const ll_t* pBeta1,
                  const ll_t* pBeta2,
                  KeyType* pAllKeys,
                  ValueType* pAllValues,
                  ll_t* pAllUUIDs,
                  ll_t numBuckets,
                  ll_t emptyMarker
                  )
                  : _pAllKeys(pAllKeys)
                  , _pAllValues(pAllValues)
                  , _pAllUUIDs(pAllUUIDs)
                  , _numBuckets(numBuckets)
                  , _emptyMarker(emptyMarker)
    {
      #pragma unroll
      for (int i=0; i < KeySize; i++){
        _prime1[i] = pPrime1[i];
        _prime2[i] = pPrime2[i];
        _alpha1[i] = pAlpha1[i];
        _alpha2[i] = pAlpha2[i];
        _beta1[i] = pBeta1[i];
        _beta2[i] = pBeta2[i];
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
    void set_key(ll_t address, KeyType key[KeySize]){
      #pragma unroll
      for (int i=0; i<KeySize; i++){
        _pAllKeys[address * KeySize + i] = key[i];
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
      ll_t hashCode = get_hash(key);
      ll_t uuid = get_uuid(key);
      #pragma unroll 2
      for (ll_t i=0; i<_numBuckets; i++){
        ll_t address = (hashCode + i) % _numBuckets;
        ll_t candidateUUID;
        bool isSuccessful = set_uuid_if_empty(address, uuid, candidateUUID);
        if (isSuccessful){
          set_key(address, key);
          set_value(address, value);
          return true;
        }
        // check if the candidateUUID is equal to uuid
        bool isFound = uuid == candidateUUID;
        // if so, return stored
        if (isFound){
          set_key(address, key);
          set_value(address, value);
          return true;
        }
      }
      return false;
    }
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
    pAllKeys,
    pAllValues,
    pAllUUIDs,
    numBuckets,
    -1
  );
  __shared__ ValueType smemValues[KPT][ValueSize][TPB];

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
    pAllKeys,
    pAllValues,
    pAllUUIDs,
    numBuckets,
    -1
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
