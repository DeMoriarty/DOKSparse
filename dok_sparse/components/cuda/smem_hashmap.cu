#define BoolType uint8_t
#define MaxIterLimit 100000
#define NULL_NODE -1

#define KEY_EMPTY 1
#define KEY_FOUND 2
#define KEY_NOT_FOUND 3
#define VALUE_STORED 4
#define VALUE_NOT_STORED 5

template <
  typename KeyType,
  typename ValueType,
  int KeyDim,
  int ValueDim,
  int NumBuckets,
  int TPB
>
class SmemHashmap {
  private:
    int64_t _alpha1[KeyDim];
    int64_t _alpha2[KeyDim];
    int64_t _beta1[KeyDim];
    int64_t _beta2[KeyDim];
    int64_t _prime1[KeyDim];
    int64_t _prime2[KeyDim];
    SmemTensor2D<KeyType, NumBuckets, KeyDim> _smemKey;
    SmemTensor2D<ValueType, NumBuckets, ValueDim> _smemValue;
    KeyType _emptyMarker;

  public:
    CUDA_DEVICE_INLINE
    SmemHashmap(const int64_t* pAlpha1,
            const int64_t* pAlpha2,
            const int64_t* pBeta1,
            const int64_t* pBeta2,
            const int64_t* pPrime1,
            const int64_t* pPrime2,
            SmemTensor2D<KeyType, NumBuckets, KeyDim> smemKey,
            SmemTensor2D<ValueType, NumBuckets, ValueDim> smemValue,
            KeyType emptyMarker
            )
            : _smemKey(smemKey)
            , _smemValue(smemValue)
            , _emptyMarker(emptyMarker)
    {
      #pragma unroll
      for (int i=0; i<KeyDim; i++){
        _alpha1[i] = pAlpha1[i];
        _alpha2[i] = pAlpha2[i];
        _beta1[i] = pBeta1[i];
        _beta2[i] = pBeta2[i];
        _prime1[i] = pPrime1[i];
        _prime2[i] = pPrime2[i];
      }

      // fill smemKey with emptyKeys
      KeyType emptyKey[KeyDim] = { _emptyMarker };
      int tid = threadIdx.x;
      #pragma unroll
      for (int i=0; i<div_ru(NumBuckets, TPB); i++){
        int address = i * TPB + tid;
        if (address < NumBuckets){
          set_key(address, emptyKey);
        }
      }
      __syncthreads();
    }

    CUDA_DEVICE_INLINE
    int64_t get_hash_1(KeyType key[KeyDim]){
      int64_t hash_code = ( (int64_t) key[0] * _alpha1[0] + _beta1[0]) % _prime1[0];
      #pragma unroll
      for (int i=1; i<KeyDim; i++){
        hash_code *= ( (int64_t) key[i] * _alpha1[i] + _beta1[i]) % _prime1[i];
      }
      hash_code = llabs(hash_code);
      hash_code = hash_code % NumBuckets;
      return hash_code;
    }

    CUDA_DEVICE_INLINE
    int64_t get_hash_2(KeyType key[KeyDim], int64_t firstHashCode){
      int64_t hash_code = ( (int64_t) key[0] * _alpha2[0] + _beta2[0]) % _prime2[0];
      #pragma unroll
      for (int i=1; i<KeyDim; i++){
        hash_code *= ( (int64_t) key[i] * _alpha2[i] + _beta2[i]) % _prime2[i];
      }
      hash_code = llabs(hash_code) + firstHashCode;
      hash_code = hash_code % NumBuckets;
      return hash_code;
    }

    CUDA_DEVICE_INLINE
    bool are_keys_equal(KeyType key1[KeyDim], KeyType key2[KeyDim]){
      bool isEqual = key1[0] == key2[0];
      #pragma unroll
      for (int i=0; i<KeyDim; i++){
        isEqual = isEqual && (key1[i] == key2[i]);
      }
      return isEqual;
    }

    CUDA_DEVICE_INLINE
    void get_key(int address, KeyType key[KeyDim]){
      #pragma unroll
      for (int i=0; i<KeyDim; i++){
        key[i] = _smemKey.get(address, i);
      }
    }

    CUDA_DEVICE_INLINE
    void set_key(int address, KeyType key[KeyDim]){
      #pragma unroll
      for (int i=0; i<KeyDim; i++){
        _smemKey.set(address, i, key[i]);
      }
    }

    CUDA_DEVICE_INLINE
    bool set_key_if_empty(int address, KeyType key[KeyDim]){
      KeyType *ptr = _smemKey.get_ptr(address, 0);
      // if the value at `ptr` is equal to `_emptyMarker`, then set the value of that pointer to `key[0]`
      // else, return `failed`
      if ( atomicCAS(ptr, _emptyMarker, key[0]) != _emptyMarker){
        return false;
      }
      #pragma unroll
      for (int i=1; i<KeyDim; i++){
        _smemKey.set(address, i, key[i]);
      }
      return true;
    }

    CUDA_DEVICE_INLINE
    void get_value(int address, ValueType value[ValueDim]){
      #pragma unroll
      for (int i=0; i<ValueDim; i++){
        value[i] = _smemValue.get(address, i);
      }
    }

    CUDA_DEVICE_INLINE
    void set_value(int address, ValueType value[ValueDim]){
      #pragma unroll
      for (int i=0; i<ValueDim; i++){
        _smemValue.set(address, i, value[i]);
      }
    }

    CUDA_DEVICE_INLINE
    int get_value_of_key(int address, KeyType key[KeyDim], ValueType value[ValueDim]){
      KeyType candidateKey[KeyDim];
      get_key(address, candidateKey);
      // check if the candidateKey is emptyKey
      bool isEmpty = candidateKey[0] == _emptyMarker;
      // is so, return not found
      if (isEmpty){
        return KEY_EMPTY;
      }
      // check if the candidateKey is equal to key
      bool isFound = are_keys_equal(candidateKey, key);
      // if so, return found
      if (isFound){
        get_value(address, value);
        return KEY_FOUND;
      }
      return KEY_NOT_FOUND;
    }

    CUDA_DEVICE_INLINE
    int set_value_of_key(int address, KeyType key[KeyDim], ValueType value[ValueDim]){
      KeyType candidateKey[KeyDim];
      get_key(address, candidateKey);
      // check if the candidateKey is emptyKey
      bool isEmpty = candidateKey[0] == _emptyMarker;
      // is so, store value in this address
      if (isEmpty){
        // set key to that address, if storing failed (because of another thread using that address ), return not stored
        bool isSuccessful = set_key_if_empty(address, key);
        if (!isSuccessful){
          return VALUE_NOT_STORED;
        }
        set_value(address, value);
        return VALUE_STORED;
      }
      // check if the candidateKey is equal to key
      bool isFound = are_keys_equal(candidateKey, key);
      // if so, return stored
      if (isFound){
        set_value(address, value);
        return VALUE_STORED;
      }
      // otherwise, return not found
      return VALUE_NOT_STORED;
    }

    CUDA_DEVICE_INLINE
    bool get(KeyType key[KeyDim], ValueType value[ValueDim]){
      int64_t hashCode1 = get_hash_1(key);
      
      // first check if key is found in address = hashCode1
      int status = get_value_of_key(hashCode1, key, value);
      // if found, return true, if that address is empty, return false, otherwise, continue
      if (status == KEY_FOUND){
        return true;
      } else if (status == KEY_EMPTY){
        return false;
      }

      // if not found, then set initial address to hashCode2,
      // then do linear probing from that point
      int64_t hashCode2 = get_hash_2(key, hashCode1);
      #pragma unroll 1
      for (int i=0; i < NumBuckets; i++){
        int64_t nextAddress = (hashCode2 + i) % NumBuckets;
        status = get_value_of_key(nextAddress, key, value);
        if (status == KEY_FOUND){
          return true;
        } else if (status == KEY_EMPTY){
          return false;
        }
      }
      return false;
    }

    CUDA_DEVICE_INLINE
    bool set(KeyType key[KeyDim], ValueType value[ValueDim]){
      int64_t hashCode1 = get_hash_1(key);
      // just like get, check if key exists at address == hashCode1
      // if the key is found, then store the value to that address, return true
      // if the key at that address is empty store value to that address, return true
      // if the key is not found, do nothing
      int status = set_value_of_key(hashCode1, key, value);
      if (status == VALUE_STORED){
        return true;
      }

      int64_t hashCode2 = get_hash_2(key, hashCode1);
      #pragma unroll 1
      for (int i=0; i < NumBuckets; i++){
        int64_t nextAddress = (hashCode2 + i) % NumBuckets;
        status = set_value_of_key(nextAddress, key, value);
        if (status == VALUE_STORED){
          return true;
        }
      }
      return false;
    }
};