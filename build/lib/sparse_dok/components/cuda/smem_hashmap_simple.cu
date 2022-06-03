#define BoolType uint8_t
#define MaxIterLimit 100000
#define NULL_NODE -1

#define KEY_EMPTY 1
#define KEY_FOUND 2
#define KEY_NOT_FOUND 3
#define VALUE_STORED 4
#define VALUE_NOT_STORED 5

#define LINEAR_PROBING 0
#define QUADRATIC_PROBING 1

template <
  typename KeyType,
  typename ValueType,
  int NumBuckets,
  int TPB,
  int ProbingMethod
>
class SmemHashmapSimple {
  private:
    int64_t _prime;
    int64_t _alpha;
    int64_t _beta;
    SmemTensor1D<KeyType, NumBuckets> _smemKey;
    SmemTensor1D<ValueType, NumBuckets> _smemValue;
    KeyType _emptyMarker;

  public:
    CUDA_DEVICE_INLINE
    SmemHashmapSimple(const int64_t* pPrime,
                      const int64_t* pAlpha,
                      const int64_t* pBeta,
                      SmemTensor1D<KeyType, NumBuckets> smemKey,
                      SmemTensor1D<ValueType, NumBuckets> smemValue,
                      KeyType emptyMarker
                      )
                      : _smemKey(smemKey)
                      , _smemValue(smemValue)
                      , _emptyMarker(emptyMarker)
    {
      _prime = pPrime[0];
      _alpha = pAlpha[0];
      _beta = pBeta[0];

      // fill smemKey with emptyKeys
      int tid = threadIdx.x;
      #pragma unroll
      for (int i=0; i<div_ru(NumBuckets, TPB); i++){
        int address = i * TPB + tid;
        if (address < NumBuckets){
          set_key(address, _emptyMarker);
        }
      }
      __syncthreads();
    }

    CUDA_DEVICE_INLINE
    int64_t get_hash(KeyType key){
      // TODO: I don't know if we can remove this _prime. because integer modulo is expensive. can we replace module with multiplication
      // and still get same uniformity? the key are suppose to be random, so maybe that somwwhat helps?
      
      // int64_t hash_code = ((int64_t) (key) % _prime);// % NumBuckets;
      // log2 doesn't exist. need to be implemented differently
      // int64_t hash_code = ( (int64_t) key * _prime) >> (64 - _LOG2NUMBUCKETS_ ); // maybe simply call this Log2NumBuckets, and calculate in python
      int64_t l = 2654435769L;
      int64_t hash_code = (key * l) ;
      hash_code = llabs(hash_code);
      return hash_code;
    }

    CUDA_DEVICE_INLINE
    bool are_keys_equal(KeyType key1, KeyType key2){
      bool isEqual = key1 == key2;
      return isEqual;
    }

    CUDA_DEVICE_INLINE
    void get_key(int address, KeyType &key){
      key = _smemKey.get(address);
    }

    CUDA_DEVICE_INLINE
    KeyType get_key(int address){
      return _smemKey.get(address);
    }

    CUDA_DEVICE_INLINE
    void set_key(int address, KeyType key){
      _smemKey.set(address, key);
    }

    CUDA_DEVICE_INLINE
    bool set_key_if_empty(int address, KeyType key){
      KeyType *ptr = _smemKey.get_ptr(address);
      // if the value at `ptr` is equal to `_emptyMarker`, then set the value of that pointer to `key[0]`
      // else, return `failed`
      if ( atomicCAS(ptr, _emptyMarker, key) != _emptyMarker){
      // if ( atomicCAS(&_smemKey.get(address), _emptyMarker, key) != _emptyMarker){
        return false;
      }
      return true;
    }

    CUDA_DEVICE_INLINE
    void get_value(int address, ValueType &value){
      value = _smemValue.get(address);
    }

    CUDA_DEVICE_INLINE
    ValueType get_value(int address){
      return _smemValue.get(address);
    }

    CUDA_DEVICE_INLINE
    void set_value(int address, ValueType value){
      _smemValue.set(address, value);
    }

    CUDA_DEVICE_INLINE
    int get_value_of_key(int address, KeyType key, ValueType &value){
      KeyType candidateKey = get_key(address);
      // check if the candidateKey is emptyKey
      bool isEmpty = candidateKey == _emptyMarker;
      // is so, return not found
      if (isEmpty){
        return KEY_EMPTY;
      }
      // check if the candidateKey is equal to key
      bool isFound = are_keys_equal(candidateKey, key);
      // if so, return found
      if (isFound){
        value = get_value(address);
        return KEY_FOUND;
      }
      return KEY_NOT_FOUND;
    }

    CUDA_DEVICE_INLINE
    int set_value_of_key(int address, KeyType key, ValueType value){
      KeyType candidateKey = get_key(address);
      // check if the candidateKey is emptyKey
      bool isEmpty = candidateKey == _emptyMarker;
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
    bool get_old(KeyType key, ValueType &value){
      // if not found, then set initial address to hashCode2,
      // then do linear probing from that point
      int64_t hashCode = get_hash(key);
      #pragma unroll 2
      for (int i=0; i < NumBuckets; i++){
        int64_t nextAddress = (hashCode + i) % NumBuckets;
        int status = get_value_of_key(nextAddress, key, value);
        if (status == KEY_FOUND){
          return true;
        } else if (status == KEY_EMPTY){
          return false;
        }
      }
      return false;
    }

    CUDA_DEVICE_INLINE
    bool get(KeyType key, ValueType &value, ValueType fallbackValue){
      // if not found, then set initial address to hashCode2,
      // then do linear probing from that point
      int64_t hashCode = get_hash(key);
      value = fallbackValue;
      #pragma unroll 1
      for (int i=0; i < NumBuckets; i++){
        // int64_t nextAddress = (hashCode + i * i) % NumBuckets; // Quadratic Probing. Num Buckets should be a power of 2.
        int64_t nextAddress = 0;
        if (ProbingMethod == LINEAR_PROBING){
          nextAddress = (hashCode + i) % NumBuckets;
        } else if (ProbingMethod == QUADRATIC_PROBING){
          nextAddress = (hashCode + i * i) % NumBuckets;
        }
        KeyType candidateKey = get_key( (int) nextAddress);
        // check if the candidateKey is emptyKey
        bool isEmpty = candidateKey == _emptyMarker;
        // is so, return not found
        if (isEmpty){
          return false;
        }
        // check if the candidateKey is equal to key
        bool isFound = candidateKey == key;
        // if so, return found
        if (isFound){
          get_value( (int) nextAddress, value);
          return true;
        }
      }
      return false;
    }

    CUDA_DEVICE_INLINE
    bool set(KeyType key, ValueType value){
      // just like get, check if key exists at address == hashCode1
      // if the key is found, then store the value to that address, return true
      // if the key at that address is empty store value to that address, return true
      // if the key is not found, do nothing
      int64_t hashCode = get_hash(key);
      #pragma unroll 3
      for (int i=0; i < NumBuckets; i++){
        int64_t nextAddress;
        if (ProbingMethod == LINEAR_PROBING){
          nextAddress = (hashCode + i) % NumBuckets;
        } else if (ProbingMethod == QUADRATIC_PROBING){
          nextAddress = (hashCode + i * i) % NumBuckets;
        }
        int status = set_value_of_key( (int) nextAddress, key, value);
        if (status == VALUE_STORED){
          return true;
        }
      }
      return false;
    }
};