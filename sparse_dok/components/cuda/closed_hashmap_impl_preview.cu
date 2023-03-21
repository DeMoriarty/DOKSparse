#include "cuda_fp16.h"
#include <stdio.h>
// #include "mma"

#define likely(x)      __builtin_expect(!!(x), 1)
#define unlikely(x)    __builtin_expect(!!(x), 0)
#define load(x)        __ldcg(x)
#define store(x, value) __stcs(x, value)
#define div_ru(a, b) (a + b - 1) / b 
#define div_rd(a, b) a / b 
#define VOLATILE
#ifndef INFINITY
#define INFINITY __int_as_float(0x7f800000)
#endif
#define DEBUG

#ifdef DEBUG
#define COMPILER_ASSERT(EXPRESSION)   switch (0) {case 0: case (EXPRESSION):;}
#else
#define COMPILER_ASSERT(EXPRESSION)
#endif

#define CUDA_DEVICE_INLINE __device__ __forceinline__

// #define LAYOUT_C true
// #define LAYOUT_F false
// #define TRANSFORM_N true
// #define TRANSFORM_T false

// typedef bool MATRIX_LAYOUT;
// typedef bool MATRIX_TRANSFORM;
typedef unsigned char uint8_t;
typedef long long ll_t;
typedef unsigned long long ull_t;

typedef struct __builtin_align__(8){
  half x1, x2, x3, x4;
} half4;

typedef struct __builtin_align__(16){
  half x1, x2, x3, x4, x5, x6, x7, x8;
} half8;

typedef struct __builtin_align__(16){
  unsigned char x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16;
} uchar16;

typedef struct __builtin_align__(16){
  char x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16;
} char16;

typedef struct {
  int x;
} Coord1D;

typedef struct {
  int x, y;
} Coord2D;

typedef struct {
  int x, y, z;
} Coord3D;

typedef struct {
  int x, y, z, t;
} Coord4D;

typedef struct {
  int x, y, z, t, u;
} Coord5D;

typedef union {
  int as_int32[1];
  unsigned int as_uint32[1];
  short as_int16[2];
  unsigned short as_uint16[2];
  signed char as_int8[4];
  unsigned char as_uint8[4]; 
  float as_float[1];
  half2 as_half2[1];
  half as_half[2];  
} Data4B;

typedef union {
  long long as_int64[1];
  unsigned long long as_uint64[1];
  int as_int32[2];
  unsigned int as_uint32[2];
  short as_int16[4];
  unsigned short as_uint16[4];
  signed char as_int8[8];
  unsigned char as_uint8[8]; 
  double as_double[1];
  half4 as_half4[1];
  float2 as_float2[1];
  float as_float[2];
  half2 as_half2[2];
  half as_half[4];  
} Data8B;

typedef union {
  uchar16 as_uchar16[1];
  char16 as_char16[1];
  long long as_int64[2];
  unsigned long long as_uint64[2];
  int as_int32[4];
  unsigned int as_uint32[4];
  short as_int16[8];
  unsigned short as_uint16[8];
  signed char as_int8[16];
  unsigned char as_uint8[16];
  half8 as_half8[1]; 
  double as_double[2];
  half4 as_half4[2];
  float2 as_float2[2];
  float as_float[4];
  half2 as_half2[4];
  half as_half[8];  
} Data16B;


template <typename MutexType>
CUDA_DEVICE_INLINE
void mutex_lock_thread(
  MutexType *mutex,
  const MutexType onValue,
  const MutexType offValue
) {
  unsigned int ns = 8;
  unsigned int counter = 0;
  while (atomicCAS(mutex, offValue, onValue) == onValue) {
    __nanosleep(ns);
    counter ++;
    if (counter > 1000) break;
    if (ns < 256) {
      ns *= 2;
    }
  }
}

template <typename MutexType>
CUDA_DEVICE_INLINE
void mutex_unlock_thread(
  MutexType *mutex,
  const MutexType offValue
) {
  __threadfence();
  atomicExch(mutex, offValue);
  __threadfence();
}

CUDA_DEVICE_INLINE
long long atomicCAS(
  ll_t *address,
  ll_t compare,
  ll_t val
){
  ull_t old = atomicCAS(
    reinterpret_cast<ull_t*>(address),
    reinterpret_cast<ull_t&>(compare),
    reinterpret_cast<ull_t&>(val)
  );
  return reinterpret_cast<ll_t&>(old);
}

CUDA_DEVICE_INLINE
long long atomicExch(
  ll_t *address,
  ll_t val
){
  ull_t old = atomicExch(
    reinterpret_cast<ull_t*>(address),
    reinterpret_cast<ull_t&>(val)
  );
  return reinterpret_cast<ll_t&>(old);
}

CUDA_DEVICE_INLINE
ll_t atomicAdd(
  ll_t *address,
  ll_t val
){
  ull_t old = atomicAdd(
    reinterpret_cast<ull_t*>(address),
    reinterpret_cast<ull_t&>(val)
  );
  return reinterpret_cast<ll_t&>(old);
}
template <
  typename T
>
class SmemTensor0D{
  public:
    VOLATILE T* endPtr;
    VOLATILE T* startPtr;

    CUDA_DEVICE_INLINE
    SmemTensor0D(VOLATILE void* smemPtr)  
        : startPtr(reinterpret_cast<T*>(smemPtr))
        , endPtr(reinterpret_cast<T*>(smemPtr))
    {
    }

    CUDA_DEVICE_INLINE
    T get(){
      return startPtr[0];
    }

    CUDA_DEVICE_INLINE
    T* get_ptr(){
      return startPtr;
    }

    CUDA_DEVICE_INLINE
    void set(T value){
      startPtr[0] = value;
    }

    template <typename U>
    CUDA_DEVICE_INLINE
    U get_reinterpreted(){
      U* newPtr = reinterpret_cast<U*>(startPtr);
      return newPtr[0];
    }

    template <typename U>
    CUDA_DEVICE_INLINE
    void set_reinterpreted(U value){
      U* newPtr = reinterpret_cast<U*>(startPtr);
      newPtr[0] = value;
    }

    template <typename U>
    CUDA_DEVICE_INLINE
    U* get_ptr_reinterpreted(){
      U* newPtr = reinterpret_cast<U*>(startPtr);
      return &newPtr[0];
    }
};

template <
  typename T,
  int ShapeX
>
class SmemTensor1D{
  public:
    VOLATILE T* endPtr;
    VOLATILE T* startPtr;

    CUDA_DEVICE_INLINE
    SmemTensor1D(VOLATILE void* smemPtr)  
        : startPtr(reinterpret_cast<T*>(smemPtr))
        , endPtr(reinterpret_cast<T*>(smemPtr) + shape().x)
    {
    }

    CUDA_DEVICE_INLINE
    T get(int x){
      return startPtr[x];
    }

    CUDA_DEVICE_INLINE
    T* get_ptr(int x){
      return &startPtr[x];
    }

    CUDA_DEVICE_INLINE
    void set(int x, T value){
      startPtr[x] = value;
    }

    template <typename U>
    CUDA_DEVICE_INLINE
    U get_reinterpreted(int x){
      U* newPtr = reinterpret_cast<U*>(startPtr);
      return newPtr[x];
    }

    template <typename U>
    CUDA_DEVICE_INLINE
    void set_reinterpreted(int x, U value){
      U* newPtr = reinterpret_cast<U*>(startPtr);
      newPtr[x] = value;
    }

    template <typename U>
    CUDA_DEVICE_INLINE
    U* get_ptr_reinterpreted(int x){
      U* newPtr = reinterpret_cast<U*>(startPtr);
      return &newPtr[x];
    }

    CUDA_DEVICE_INLINE
    static constexpr Coord1D shape(){
      return { ShapeX };
    }
};

template <
  typename T,
  int ShapeX,
  int ShapeY
>
class SmemTensor2D{
  public:
    VOLATILE T* endPtr;
    VOLATILE T* startPtr;

    CUDA_DEVICE_INLINE
    SmemTensor2D(VOLATILE void* smemPtr)  
        : startPtr(reinterpret_cast<T*>(smemPtr))
        , endPtr(reinterpret_cast<T*>(smemPtr) + shape().x * shape().y)
    {
    }

    CUDA_DEVICE_INLINE
    T get(int x, int y){
      return startPtr[x * stride().x + y];
    }

    CUDA_DEVICE_INLINE
    T* get_ptr(int x, int y){
      return &startPtr[x * stride().x + y];
    }

    CUDA_DEVICE_INLINE
    void set(int x, int y, T value){
      startPtr[x * stride().x + y] = value;
    }

    CUDA_DEVICE_INLINE
    SmemTensor1D<T, ShapeY> get_child(int x){
      SmemTensor1D<T, ShapeY> child(
        &startPtr[x * stride().x]
      );
      return child;
    }

    template <typename U>
    CUDA_DEVICE_INLINE
    U get_reinterpreted(int x, int y){
      U* newPtr = reinterpret_cast<U*>(startPtr);
      return newPtr[
        (x * stride().x) * sizeof(T) / sizeof(U) + 
        y
      ];
    }

    template <typename U>
    CUDA_DEVICE_INLINE
    void set_reinterpreted(int x, int y, U value){
      U* newPtr = reinterpret_cast<U*>(startPtr);
      newPtr[
        (x * stride().x) * sizeof(T) / sizeof(U) + 
        y
      ] = value;
    }

    template <typename U>
    CUDA_DEVICE_INLINE
    U* get_ptr_reinterpreted(int x, int y){
      U* newPtr = reinterpret_cast<U*>(startPtr);
      return &newPtr[
        (x * stride().x) * sizeof(T) / sizeof(U) + 
        y
      ];
    }

    CUDA_DEVICE_INLINE
    static constexpr Coord2D shape(){
      return {ShapeX, ShapeY};
    }

    CUDA_DEVICE_INLINE
    static constexpr Coord1D stride(){
      return {ShapeY};
    }

};

template <
  typename T,
  int ShapeX,
  int ShapeY,
  int ShapeZ
>
class SmemTensor3D{
  public:
    VOLATILE T* endPtr;
    VOLATILE T* startPtr;

    CUDA_DEVICE_INLINE
    SmemTensor3D(VOLATILE void* smemPtr)  
        : startPtr(reinterpret_cast<T*>(smemPtr))
        , endPtr(reinterpret_cast<T*>(smemPtr) + shape().x * shape().y * shape().z)
    {
    }

    CUDA_DEVICE_INLINE
    T get(int x, int y, int z){
      return startPtr[x * stride().x + y * stride().y + z];
    }

    CUDA_DEVICE_INLINE
    T* get_ptr(int x, int y, int z){
      return &startPtr[x * stride().x + y * stride().y + z];
    }

    CUDA_DEVICE_INLINE
    void set(int x, int y, int z, T value){
      startPtr[x * stride().x + y * stride().y + z] = value;
    }

    CUDA_DEVICE_INLINE
    SmemTensor2D<T, ShapeY, ShapeZ> get_child(int x){
      SmemTensor2D<T, ShapeY, ShapeZ> child(
        &startPtr[x * stride().x]
      );
      return child;
    }

    CUDA_DEVICE_INLINE
    SmemTensor1D<T, ShapeZ> get_child(int x, int y){
      SmemTensor1D<T, ShapeZ> child(
        &startPtr[x * stride().x + y * stride().y]
      );
      return child;
    }

    template <typename U>
    CUDA_DEVICE_INLINE
    U get_reinterpreted(int x, int y, int z){
      U* newPtr = reinterpret_cast<U*>(startPtr);
      return newPtr[
        (x * stride().x +  
        y * stride().y) * sizeof(T) / sizeof(U) + 
        z
      ];
    }

    template <typename U>
    CUDA_DEVICE_INLINE
    void set_reinterpreted(int x, int y, int z, U value){
      U* newPtr = reinterpret_cast<U*>(startPtr);
      newPtr[
        (x * stride().x +  
        y * stride().y) * sizeof(T) / sizeof(U) + 
        z
      ] = value;
    }

    template <typename U>
    CUDA_DEVICE_INLINE
    U* get_ptr_reinterpreted(int x, int y, int z){
      U* newPtr = reinterpret_cast<U*>(startPtr);
      return &newPtr[
        (x * stride().x +  
        y * stride().y) * sizeof(T) / sizeof(U) + 
        z
      ];
    }

    CUDA_DEVICE_INLINE
    static constexpr Coord3D shape(){
      return {ShapeX, ShapeY, ShapeZ};
    }

    CUDA_DEVICE_INLINE
    static constexpr Coord2D stride(){
      return {ShapeY * ShapeZ, ShapeZ};
    }

};

template <
  typename T,
  int ShapeX,
  int ShapeY,
  int ShapeZ,
  int ShapeT
>
class SmemTensor4D{
  public:
    VOLATILE T* endPtr;
    VOLATILE T* startPtr;
    // const Coord3D _stride;
    // const Coord4D _shape;

    CUDA_DEVICE_INLINE
    SmemTensor4D(VOLATILE void* smemPtr)  
        : startPtr(reinterpret_cast<T*>(smemPtr))
        , endPtr(&reinterpret_cast<T*>(smemPtr)[shape().x * shape().y * shape().z * shape().t])
    {
    }

    CUDA_DEVICE_INLINE
    T get(int x, int y, int z, int t){
      return startPtr[
        x * stride().x + 
        y * stride().y + 
        z * stride().z +
        t
      ];
    }

    CUDA_DEVICE_INLINE
    T* get_ptr(int x, int y, int z, int t){
      return &startPtr[
        x * stride().x + 
        y * stride().y + 
        z * stride().z +
        t
      ];
    }

    CUDA_DEVICE_INLINE
    void set(int x, int y, int z, int t, T value){
      startPtr[
        x * stride().x + 
        y * stride().y + 
        z * stride().z +
        t
      ] = value;
    }

    CUDA_DEVICE_INLINE
    SmemTensor3D<T, ShapeY, ShapeZ, ShapeT> get_child(int x){
      SmemTensor3D<T, ShapeY, ShapeZ, ShapeT> child(
        &startPtr[x * stride().x]
      );
      return child;
    }

    CUDA_DEVICE_INLINE
    SmemTensor2D<T, ShapeZ, ShapeT> get_child(int x, int y){
      SmemTensor2D<T, ShapeZ, ShapeT> child(
        &startPtr[x * stride().x + y * stride().y]
      );
      return child;
    }

    CUDA_DEVICE_INLINE
    SmemTensor1D<T, ShapeT> get_child(int x, int y, int z){
      SmemTensor1D<T, ShapeT> child(
        &startPtr[x * stride().x + y * stride().y + z * stride().z]
      );
      return child;
    }

    template <typename U>
    CUDA_DEVICE_INLINE
    U get_reinterpreted(int x, int y, int z, int t){
      U* newPtr = reinterpret_cast<U*>(startPtr);
      return newPtr[
        (x * stride().x +  
        y * stride().y +  
        z * stride().z) * sizeof(T) / sizeof(U) + 
        t
      ];
    }

    template <typename U>
    CUDA_DEVICE_INLINE
    void set_reinterpreted(int x, int y, int z, int t, U value){
      U* newPtr = reinterpret_cast<U*>(startPtr);
      newPtr[
        (x * stride().x +  
        y * stride().y +  
        z * stride().z) * sizeof(T) / sizeof(U) + 
        t
      ] = value;
    }

    template <typename U>
    CUDA_DEVICE_INLINE
    U* get_ptr_reinterpreted(int x, int y, int z, int t){
      U* newPtr = reinterpret_cast<U*>(startPtr);
      return &newPtr[
        (x * stride().x +  
        y * stride().y +  
        z * stride().z) * sizeof(T) / sizeof(U) + 
        t
      ];
    }

    CUDA_DEVICE_INLINE
    static constexpr Coord4D shape(){
      return {ShapeX, ShapeY, ShapeZ, ShapeT};
    }

    CUDA_DEVICE_INLINE
    static constexpr Coord3D stride(){
      return {
        ShapeY * ShapeZ * ShapeT, 
        ShapeZ * ShapeT, 
        ShapeT
      };
    }
};
#define EMPTY 1
#define FOUND 2
#define NOT_FOUND 3
#define STORED 4
#define NOT_STORED 5

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
      // _numElements = pArgs[KeySize * 7 + 4]; //FIXME: useless for now
      _emptyMarker = pArgs[KeySize * 7 + 4];
      _removedMarker = pArgs[KeySize * 7 + 5];
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
    bool set_uuid_if_empty(int address, ll_t uuid, ll_t &oldUUID){
      ll_t *ptr = &_pAllUUIDs[address];
      // if the value at `ptr` is equal to `_emptyMarker`, then set the value of that pointer to `uuid`, return true
      // else, return false
      __threadfence();
      oldUUID = atomicCAS(ptr, _emptyMarker, uuid);
      __threadfence();
      if ( oldUUID == _emptyMarker){
        return true;
      }
      return false;
    }

    CUDA_DEVICE_INLINE
    bool set_uuid_if_removed(int address, ll_t uuid, ll_t &oldUUID){
      ll_t *ptr = &_pAllUUIDs[address];
      // if the value at `ptr` is equal to `_removedMarker`, then set the value of that pointer to `uuid`, return true
      // else, return false
      __threadfence();
      oldUUID = atomicCAS(ptr, _removedMarker, uuid);
      __threadfence();
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
    int set_by_uuid(int address, ll_t uuid, key_t key[KeySize], value_t value[ValueSize]){
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
    bool exists(
      key_t key[KeySize]
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
          return true;
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
    bool set_v0(
      key_t key[KeySize],
      value_t value[ValueSize]
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
    bool set(
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
          if (firstRemovedAddress == -1){
            if (set_uuid_if_empty(address, uuid, candidateUUID)){
              set_key_permuted(address, key);
              set_value(address, value);
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
          if (set_uuid_if_removed(address, uuid, candidateUUID)){
            set_key_permuted(address, key);
            set_value(address, value);
            return true;
          } else if (set_uuid_if_empty(address, uuid, candidateUUID)){
            set_key_permuted(address, key);
            set_value(address, value);
            return true;
          }
        }
      }
      return false;
    }

    CUDA_DEVICE_INLINE
    bool set_old(
      key_t key[KeySize],
      value_t value[ValueSize]
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
      key_t key[KeySize]
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
      key_t key[BatchSize][KeySize],
      value_t value[BatchSize][ValueSize],
      value_t fallbackValue[ValueSize],
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
      key_t key[BatchSize][KeySize],
      value_t value[BatchSize][ValueSize],
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