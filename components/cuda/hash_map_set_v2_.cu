#include "cuda_fp16.h"

#define likely(x)      __builtin_expect(!!(x), 1)
#define unlikely(x)    __builtin_expect(!!(x), 0)
#define div_ru(a, b) (a + b - 1) / b 
#define div_rd(a, b) a / b 
#define VOLATILE
#define NodeType int64_t
#define MutexType int
#define KeyType _KEYTYPE_
#define MaxIterLimit 100000
#define NULL_NODE -1

#if (__CUDA_ARCH__ < 700)
__device__ void __nanosleep(unsigned int ns){
  clock_t start_clock = clock();
  clock_t clock_offset = 0;
  while (clock_offset < ns)
  {
    clock_offset = clock() - start_clock;
  }
}
#endif 

__device__ __forceinline__
void mutex_lock_block(
  MutexType *mutex
) {
  unsigned int ns = 8;
  unsigned int counter = 0;
  __syncthreads();
  if (threadIdx.x == 0 ){
    while (atomicCAS(mutex, 0, 1) == 1) {
      __nanosleep(ns);
      counter ++;
      if (counter > 1000) break;
      if (ns < 256) {
        ns *= 2;
      }
    }
  }
  __syncthreads();
}

__device__ __forceinline__
void mutex_unlock_block(
  MutexType *mutex
) {
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0){
    atomicExch(mutex, 0);
    __threadfence();
  }
  __syncthreads();
}

__device__ __forceinline__
void mutex_lock_thread(
  MutexType *mutex
) {
  unsigned int ns = 8;
  unsigned int counter = 0;
  while (atomicCAS(mutex, 0, 1) == 1) {
    __nanosleep(ns);
    counter ++;
    if (counter > 1000) break;
    if (ns < 256) {
      ns *= 2;
    }
  }
}

__device__ __forceinline__
void mutex_unlock_thread(
  MutexType *mutex
) {
  __threadfence();
  atomicExch(mutex, 0);
  __threadfence();
}

__device__ __forceinline__
void all_true_32(
  bool &value
){
  constexpr int width = 32;
  const int wx = threadIdx.x % 32;
  int ivalue = (int) value;
  ivalue &= __shfl_down_sync(0xFFFFFFFF, ivalue, 16);
  ivalue &= __shfl_down_sync(0xFFFFFFFF, ivalue, 8);
  ivalue &= __shfl_down_sync(0xFFFFFFFF, ivalue, 4);
  ivalue &= __shfl_down_sync(0xFFFFFFFF, ivalue, 2);
  ivalue &= __shfl_down_sync(0xFFFFFFFF, ivalue, 1);
  ivalue = __shfl_sync(0xFFFFFFFF, ivalue, wx / width);
  value = (bool) ivalue;
}

__device__ __forceinline__
void all_true_16(
  bool &value
){
  constexpr int width = 16;
  const int wx = threadIdx.x % 32;
  int ivalue = (int) value;
  ivalue &= __shfl_down_sync(0xFFFFFFFF, ivalue, 8);
  ivalue &= __shfl_down_sync(0xFFFFFFFF, ivalue, 4);
  ivalue &= __shfl_down_sync(0xFFFFFFFF, ivalue, 2);
  ivalue &= __shfl_down_sync(0xFFFFFFFF, ivalue, 1);
  ivalue = __shfl_sync(0xFFFFFFFF, ivalue, wx / width);
  value = (bool) ivalue;
}

__device__ __forceinline__
void all_true_8(
  bool &value
){
  constexpr int width = 8;
  const int wx = threadIdx.x % 32;
  int ivalue = (int) value;
  ivalue &= __shfl_down_sync(0xFFFFFFFF, ivalue, 4);
  ivalue &= __shfl_down_sync(0xFFFFFFFF, ivalue, 2);
  ivalue &= __shfl_down_sync(0xFFFFFFFF, ivalue, 1);
  ivalue = __shfl_sync(0xFFFFFFFF, ivalue, wx / width);
  value = (bool) ivalue;
}

__device__ __forceinline__
void all_true_4(
  bool &value
){
  constexpr int width = 4;
  const int wx = threadIdx.x % 32;
  int ivalue = (int) value;
  ivalue &= __shfl_down_sync(0xFFFFFFFF, ivalue, 2);
  ivalue &= __shfl_down_sync(0xFFFFFFFF, ivalue, 1);
  ivalue = __shfl_sync(0xFFFFFFFF, ivalue, wx / width);
  value = (bool) ivalue;
}

__device__ __forceinline__
void all_true_2(
  bool &value
){
  constexpr int width = 2;
  const int wx = threadIdx.x % 32;
  int ivalue = (int) value;
  ivalue &= __shfl_down_sync(0xFFFFFFFF, ivalue, 1);
  ivalue = __shfl_sync(0xFFFFFFFF, ivalue, wx / width);
  value = (bool) ivalue;
}

__device__ __forceinline__
void all_true_1(
  bool &value
){
}


template <int width>
__device__ __forceinline__
void all_true(
  bool &value
){
  // if (width == 1){
  //   all_true_1(value);
  // } else if (width == 2)
  switch (width){
    case 1:
      all_true_1(value); break;
    
    case 2:
      all_true_2(value); break;
    
    case 4:
      all_true_4(value); break;
    
    case 8:
      all_true_8(value); break;
    
    case 16:
      all_true_16(value); break;
    
    case 32:
      all_true_32(value); break;
  }
}


extern "C"
__global__ void hash_map_set_p1(
  const KeyType* __restrict__ pNewKey,    //[numNewKeys, KeySize]
  const NodeType* __restrict__ pHashCode, //[numNewKeys]
  const KeyType* __restrict__ pKey,      // [numKeys, KeySize]
  const NodeType* __restrict__ pNextNode, //[numKeys]
  uint8_t* __restrict__ pIsHeadNodeEmpty, //[numBuckets]
  NodeType* pLastVisitedNodeStatus,//[numNewKeys]
  uint8_t* pIsKeyFound,     //[numNewKeys]
  const int numNewKeys
){
  constexpr int ThreadsPerBlock = _TPB_;
  constexpr int numNewKeysPerBlock = _KPB_;
  constexpr int WarpSize = 32;
  constexpr int WarpCount = ThreadsPerBlock / WarpSize;
  constexpr int KeySize = _KEYSIZE_;
  constexpr int KeySizePo2 = _KEYSIZEPO2_;
  constexpr int MaxKeysPerWarpPerIter = max(WarpSize / KeySizePo2, 1);
  constexpr int MaxKeysPerIter = MaxKeysPerWarpPerIter * WarpCount;
  constexpr int NumIters = div_ru(numNewKeysPerBlock, MaxKeysPerIter);

  const int tid = threadIdx.x;
  const int wx = tid % WarpSize;
  const int wy = tid / WarpSize;
  const int gKeyIdxStart = blockIdx.x * numNewKeysPerBlock;

  extern __shared__ KeyType newKeySmem[]; //[numNewKeysPerBlock * KeySize]

  if (KeySize < 32){
    int keyOffset = tid % KeySizePo2;
    // Load newKey
    #pragma unroll
    for (int i=0; i<NumIters; i++){
      int keyIdx = i * MaxKeysPerIter + tid / KeySizePo2;
      int smemOffset = keyIdx * KeySize + keyOffset;
      int globalOffset = (gKeyIdxStart + keyIdx) * KeySize + keyOffset;
      if (keyOffset < KeySize && keyIdx < numNewKeysPerBlock && (gKeyIdxStart + keyIdx) < numNewKeys){
        newKeySmem[smemOffset] = pNewKey[globalOffset];
      }
    }
    __syncthreads();

    // Load hashCode
    NodeType currentNode[NumIters];
    // bool isHeadNodeEmpty[NumIters];
    #pragma unroll
    for (int i=0; i<NumIters; i++){
      int keyIdx = i * MaxKeysPerIter + tid / KeySizePo2;
      int gKeyIdx = gKeyIdxStart + keyIdx;
      currentNode[i] = pHashCode[gKeyIdxStart + keyIdx];
      bool isHeadNodeEmpty = (bool) pIsHeadNodeEmpty[currentNode[i]];
      if (isHeadNodeEmpty){
        pLastVisitedNodeStatus[gKeyIdx] = currentNode[i];
        pIsKeyFound[gKeyIdx] = 2;
        pIsHeadNodeEmpty[currentNode[i]] = (uint8_t) false;
      }
    }

    


  } else {
    // #pragma unroll
    // for (int i=0; i<NumIters; i++){
    //   int keyIdx = i * MaxKeysPerIter + wx;
    //   #pragma unroll
    //   for (int j=0; j < div_ru(KeySize, WarpSize); j++){
    //     int keyOffset = j * WarpSize + wx;
    //     int smemOffset = (keyIdx) * KeySize + keyOffset;
    //     int globalOffset = (gKeyIdxStart + keyIdx) * KeySize + keyOffset;
    //     if (keyOffset < KeySize && keyIdx < numNewKeysPerBlock && (gKeyIdxStart + keyIdx) < numNewKeys){
    //       newKeySmem[smemOffset] = pNewKey[globalOffset];
    //     }
    //   }
    // }
  }

  __syncthreads();
  // Load hash code as current node index
  NodeType currentNode = pHashCode[bid];
  
  // check if the head node is empty, if so, return.
  bool isHeadNodeEmpty = (bool) pIsHeadNodeEmpty[currentNode];
  if (isHeadNodeEmpty){
    // although key isn't found, but since it's a head node, we don't need to create a new node, so set isKeyFound to 2
    pLastVisitedNodeStatus[bid] = currentNode;
    pIsKeyFound[bid] = 2;
    pIsHeadNodeEmpty[currentNode] = (uint8_t) false;
    return;
  }

  int counter = 0;
  while (counter < MaxIterLimit){
    // check if the newKey is found in current node
    bool isFound;
    #pragma unroll
    for (int i=0; i<nLoadIters; i++){
      KeyType key;
      // isEqual is true by default, so even when (KeySize % ThreadPerBlock != 0), we can still compare key and newKey
      bool isEqual = true;
      int offset = i * nLoadIters + tid;
      int globalOffset = currentNode * KeySize + offset;
      if (offset < KeySize){
        // load the key in current node
        key = pKey[globalOffset]; // FIXME: this stalls
        // Compare with newKey
        isEqual = key == newKeySmem[offset];
      }
      //all reduce
      if (KeySize == 1){
        all_true<1>(isEqual);
      } else if (KeySize <= 2){
        all_true<2>(isEqual);
      } else if (KeySize <= 4){
        all_true<4>(isEqual);
      } else if (KeySize <= 8){
        all_true<8>(isEqual);
      } else if (KeySize <= 16){
        all_true<16>(isEqual);
      } else {
        all_true<32>(isEqual);
      }

      if (i == 0){
        isFound = isEqual;
      } else {
        isFound = isFound && isEqual;
      }

      // if (!isFound){
      //   break;
      // }
    }

    if (isFound){
      // if key is found, set last visited node to current node, set isKeyFound to 1
      pLastVisitedNodeStatus[bid] = currentNode;
      pIsKeyFound[bid] = 1;
      break;
    } else {
      // otherwise, check if next node exists.
      NodeType nextNode = pNextNode[currentNode];
      if (nextNode == NULL_NODE){
        // if next node doesn't exist, set last visited node to current node, set isKeyFound to 0
        pLastVisitedNodeStatus[bid] = currentNode;
        pIsKeyFound[bid] = 0;

      } else {
        // otherwise, set current node to next node.
        currentNode = nextNode;
      }
    }
    counter ++;
  }
  /*
  */
}

extern "C"
__global__ void hash_map_set_p2(
  const NodeType* __restrict__ pNewNode, //[numNewNodes]
  const NodeType* __restrict__ pLastVisitedNodeStatus, //[numNewNodes]
  NodeType* pNextNode,  //[numTotalNodes]
  int64_t* pInverse, //[numNewNodes]
  MutexType* pMutex //[numUniqueLastVisistedNode]
){
  constexpr int ThreadsPerBlock = _TPB_;
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  // lock 
  NodeType lastVisitedNode = pLastVisitedNodeStatus[bid];
  NodeType newNode = pNewNode[bid];
  MutexType* pCurrentMutex = &pMutex[pInverse[bid]];

  mutex_lock_block(pCurrentMutex);
  NodeType currentNode = lastVisitedNode;

  int counter = 0;
  // traverse the linked list untill finding the last node, and point its next node to new node.
  while (counter < MaxIterLimit){
    NodeType nextNode = pNextNode[currentNode];
    if (nextNode == NULL_NODE){
      // if the next node is null, set the next node to new node, break
      pNextNode[currentNode] = newNode;
      break;
    } else {
      // otherwise, proceed to next node
      currentNode = nextNode;
    }
    counter ++;
  }
  mutex_unlock_block(pCurrentMutex);
}
