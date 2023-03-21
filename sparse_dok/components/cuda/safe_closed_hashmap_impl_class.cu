#define EMPTY 1
#define FOUND 2
#define NOT_FOUND 3
#define STORED 4
#define NOT_STORED 5
#define MAX_STALL_ITERS 10000

template <
  typename key_t,
  typename value_t,
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
    key_t *_pAllKeys;
    value_t *_pAllValues;
    ll_t *_pAllUUIDs;
    ll_t _numBuckets;
    // ll_t _numElements;
    ll_t _emptyMarker;
    ll_t _occupiedMarker;
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
                  key_t* pAllKeys,
                  value_t* pAllValues,
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
                  , _occupiedMarker(-2) //FIXME:
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
    ClosedHashmap(const ll_t* pArgs){
      #pragma unroll
      for (int i=0; i < KeySize; i++){
        _prime1[i] = pArgs[i];
        _prime2[i] = pArgs[i + KeySize * 1];
        _alpha1[i] = pArgs[i + KeySize * 2];
        _alpha2[i] = pArgs[i + KeySize * 3];
        _beta1[i] = pArgs[i + KeySize * 4];
        _beta2[i] = pArgs[i + KeySize * 5];
        keyPerm[i] = pArgs[i + KeySize * 6];
      }
      _pAllKeys = reinterpret_cast<key_t*>(pArgs[KeySize * 7]);
      _pAllValues = reinterpret_cast<value_t*>(pArgs[KeySize * 7 + 1]);
      _pAllUUIDs = reinterpret_cast<ll_t*>(pArgs[KeySize * 7 + 2]);
      _numBuckets = pArgs[KeySize * 7 + 3];
      _emptyMarker = pArgs[KeySize * 7 + 4];
      _removedMarker = pArgs[KeySize * 7 + 5];
      _occupiedMarker = -2; // FIXME: 
    }

    CUDA_DEVICE_INLINE
    ll_t get_hash(key_t key[KeySize]){
      ll_t hash_code = ( (ll_t) key[0] * _alpha1[0] + _beta1[0]) % _prime1[0];
      #pragma unroll
      for (int i=1; i<KeySize; i++){
        hash_code *= ( (ll_t) key[i] * _alpha1[i] + _beta1[i]) % _prime1[i];
      }
      hash_code = llabs(hash_code);
      return hash_code;
    }

    CUDA_DEVICE_INLINE
    ll_t get_uuid(key_t key[KeySize]){
      ll_t uuid = ( (ll_t) key[0] * _alpha2[0] + _beta2[0]) % _prime2[0];
      #pragma unroll
      for (int i=1; i<KeySize; i++){
        uuid *= ( (ll_t) key[i] * _alpha2[i] + _beta2[i]) % _prime2[i];
      }
      uuid = llabs(uuid);
      return uuid;
    }

    CUDA_DEVICE_INLINE
    bool are_keys_equal(key_t key1[KeySize], key_t key2[KeySize]){
      bool isEqual = key1[0] == key2[0];
      #pragma unroll
      for (int i=0; i<KeySize; i++){
        isEqual = isEqual && (key1[i] == key2[i]);
      }
      return isEqual;
    }

    CUDA_DEVICE_INLINE
    void get_key(ll_t address, key_t key[KeySize]){
      #pragma unroll
      for (int i=0; i<KeySize; i++){
        key[i] = _pAllKeys[address * KeySize + i];
      }
    }

    CUDA_DEVICE_INLINE
    void get_key_permuted(ll_t address, key_t key[KeySize]){
      #pragma unroll
      for (int i=0; i<KeySize; i++){
        key[i] = _pAllKeys[address * KeySize + keyPerm[i]];
      }
    }

    CUDA_DEVICE_INLINE
    void set_key(ll_t address, key_t key[KeySize]){
      #pragma unroll
      for (int i=0; i<KeySize; i++){
        _pAllKeys[address * KeySize + i] = key[i];
      }
    
    }

    CUDA_DEVICE_INLINE
    void set_key_permuted(ll_t address, key_t key[KeySize]){
      #pragma unroll
      for (int i=0; i<KeySize; i++){
        _pAllKeys[address * KeySize + keyPerm[i] ] = key[i];
      }
    }

    CUDA_DEVICE_INLINE
    void get_value(ll_t address, value_t value[ValueSize]){
      #pragma unroll
      for (int i=0; i<ValueSize; i++){
        value[i] = _pAllValues[address * ValueSize + i];
      }
    }

    CUDA_DEVICE_INLINE
    void set_value(ll_t address, value_t value[ValueSize]){
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
    bool set_uuid_if_empty(ll_t address, ll_t uuid, ll_t &oldUUID){
      ll_t *ptr = &_pAllUUIDs[address];
      // if the value at `ptr` is equal to `_emptyMarker`, then set the value of that pointer to `uuid`, return true
      // else, return false
      oldUUID = atomicCAS(ptr, _emptyMarker, uuid);
      if ( oldUUID == _emptyMarker){
        return true;
      }
      return false;
    }

    CUDA_DEVICE_INLINE
    bool set_uuid_if_removed(ll_t address, ll_t uuid, ll_t &oldUUID){
      ll_t *ptr = &_pAllUUIDs[address];
      // if the value at `ptr` is equal to `_removedMarker`, then set the value of that pointer to `uuid`, return true
      // else, return false
      oldUUID = atomicCAS(ptr, _removedMarker, uuid);
      if ( oldUUID == _removedMarker){
        return true;
      }
      return false;
    }

    CUDA_DEVICE_INLINE
    int get_by_uuid(ll_t address, ll_t uuid, value_t value[ValueSize]){
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
    int set_by_uuid(ll_t address, ll_t uuid, key_t key[KeySize], value_t value[ValueSize]){
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
    void lock_address(ll_t address, ll_t& uuid){
      for (int c=0; c<MAX_STALL_ITERS; c++){
        __threadfence();
        uuid = atomicExch(_pAllUUIDs + address, _occupiedMarker);
        __threadfence();
        if (uuid != _occupiedMarker){
          break;
        }
        __nanosleep(100);
      }
    }

    CUDA_DEVICE_INLINE
    bool unlock_address(ll_t address, ll_t uuid){
        __threadfence();
      ll_t old = atomicCAS(_pAllUUIDs + address, _occupiedMarker, uuid);
        __threadfence();
      return old == _occupiedMarker;
    }

    CUDA_DEVICE_INLINE
    bool exists(
      key_t key[KeySize]
    ){
      // permute_key(key);
      ll_t startAddress = get_hash(key);
      ll_t uuid = get_uuid(key);
      #pragma unroll 2
      for (ll_t i=0; i < _numBuckets; i++){
        ll_t address = (startAddress + i) % _numBuckets;
        ll_t candidateUUID = get_uuid(address);

        // check if the candidateKey is equal to key
        bool isFound = candidateUUID == uuid;
        // if so, return found
        if (isFound){
          return true;
        }

        // check if the candidateKey is emptyKey
        bool isEmpty = candidateUUID == _emptyMarker;
        // is so, return not found
        if (isEmpty){
          break;
        }
      }
      return false;
    }

    CUDA_DEVICE_INLINE
    bool get(
      key_t key[KeySize],
      value_t value[ValueSize],
      value_t fallbackValue[ValueSize]
    ){
      // permute_key(key);
      ll_t startAddress = get_hash(key);
      ll_t uuid = get_uuid(key);
      #pragma unroll 2
      for (ll_t i=0; i < _numBuckets; i++){
        ll_t address = (startAddress + i) % _numBuckets;
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
    bool set_v0(
      key_t key[KeySize],
      value_t value[ValueSize]
    ){
      // permute_key(key);
      ll_t startAddress = get_hash(key);
      ll_t uuid = get_uuid(key);
      ll_t firstRemovedAddress = -1;
      #pragma unroll 2
      for (ll_t i=0; i<_numBuckets; i++){
        ll_t address = (startAddress + i) % _numBuckets;
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
    bool set(
      key_t key[KeySize],
      value_t value[ValueSize]
    ){
      // permute_key(key);
      ll_t startAddress = get_hash(key) % _numBuckets;
      ll_t uuid = get_uuid(key);
      ll_t firstRemovedAddress = -1;
      ll_t startUUID;
      lock_address(startAddress, startUUID);
      if ( (startUUID == uuid) || (startUUID == _emptyMarker)){
        set_key_permuted(startAddress, key);
        set_value(startAddress, value);
        set_uuid(startAddress, uuid);
        return true;
      } else if (startUUID == _removedMarker){
        firstRemovedAddress = startAddress;
      }
      #pragma unroll 2
      for (ll_t i=1; i<_numBuckets; i++){
        ll_t address = (startAddress + i) % _numBuckets;
        ll_t candidateUUID;
        candidateUUID = get_uuid(address);
        bool isFound = candidateUUID == uuid;
        // if key is found, return stored
        if (isFound){
          set_key_permuted(address, key);
          set_value(address, value);
          unlock_address(startAddress, startUUID);
          return true;
        }

        bool isRemoved = candidateUUID == _removedMarker;
        if (isRemoved && firstRemovedAddress == -1){
          firstRemovedAddress = address;
        }

        bool isEmpty = candidateUUID == _emptyMarker;
        if (isEmpty){
          if (firstRemovedAddress == -1){
            if (set_uuid_if_empty(address, uuid, candidateUUID)){
              set_key_permuted(address, key);
              set_value(address, value);
              unlock_address(startAddress, startUUID);
              return true;
            }
          } else {
            break;
          }
        }
      }

      if (firstRemovedAddress != -1){
        #pragma unroll 2
        for (ll_t i=0; i<_numBuckets; i++){
          ll_t address = (firstRemovedAddress + i) % _numBuckets;
          ll_t candidateUUID;
          // candidateUUID = get_uuid(address);
          if (address == startAddress){
            set_key_permuted(address, key);
            set_value(address, value);
            set_uuid(startAddress, uuid);
          } else {
            if (set_uuid_if_removed(address, uuid, candidateUUID)){
              set_key_permuted(address, key);
              set_value(address, value);
              unlock_address(startAddress, startUUID);
              return true;
            } else if (set_uuid_if_empty(address, uuid, candidateUUID)){
              set_key_permuted(address, key);
              set_value(address, value);
              unlock_address(startAddress, startUUID);
              return true;
            }
          }
        }
      }
      unlock_address(startAddress, startUUID);
      return false;
    }

    CUDA_DEVICE_INLINE
    bool remove(
      key_t key[KeySize]
    ){
      ll_t startAddress = get_hash(key);
      ll_t uuid = get_uuid(key);
      #pragma unroll 2
      for (ll_t i=0; i < _numBuckets; i++){
        ll_t address = (startAddress + i) % _numBuckets;
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
};