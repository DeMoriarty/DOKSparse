#include "cuda_fp16.h"
#include <stdio.h>

#define likely(x)      __builtin_expect(!!(x), 1)
#define unlikely(x)    __builtin_expect(!!(x), 0)
#define div_ru(a, b) (a + b - 1) / b 
#define div_rd(a, b) a / b 
#define VOLATILE
#define NodeType int64_t
#define MutexType int*
#define KeyType _KEYTYPE_
#define BoolType uint8_t
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
  MutexType mutex
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
  MutexType mutex
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
  MutexType mutex
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
  MutexType mutex
) {
  __threadfence();
  atomicExch(mutex, 0);
  __threadfence();
}

extern "C"
__global__ void hash_map_set_p1(
  const NodeType* __restrict__ pNewHashCode, //[numNew]
  const NodeType* __restrict__ pNewUUID,     //[numNew]
  const NodeType* __restrict__ pUUID,        //[numAll]
  const NodeType* __restrict__ pNextNode,    //[numAll]
  // BoolType* pIsHeadNodeEmpty,           //[numBuckets]
  NodeType* pLastVisitedNode,           //[numNew]
  uint8_t* pLastVisitedNodeStatus,       //numNew
  int numNew
){
  constexpr int TPB = _TPB_;
  constexpr int Stages = _STAGES_;
  constexpr int NewPairPerBlock = TPB * Stages;

  const int tid = threadIdx.x; 
  const int bStart = blockIdx.x * NewPairPerBlock;

  NodeType currentNode[Stages];
  NodeType newUUID[Stages];
  bool isDone[Stages] = { false };

  #pragma unroll
  for (int i=0; i<Stages; i++){
    int bid = bStart + i * TPB + tid;
    if (bid < numNew){
      currentNode[i] = pNewHashCode[bid];
      newUUID[i] = pNewUUID[bid];
      // bool isHeadNodeEmpty = pIsHeadNodeEmpty[currentNode[i]];
      // if (isHeadNodeEmpty){
      //   pLastVisitedNode[bid] = currentNode[i];
      //   pLastVisitedNodeStatus[bid] = 2;
      //   pIsHeadNodeEmpty[currentNode[i]] = (BoolType) false;
      //   isDone[i] = true;
      // }
    } else {
      isDone[i] = true;
    }
  }

  int counter = 0;
  while (counter < MaxIterLimit){
    NodeType UUIDBuffer[Stages];
    NodeType nextNodeBuffer[Stages];
    #pragma unroll
    for (int i = 0; i<Stages; i++){
      if (!isDone[i]){
        UUIDBuffer[i] = pUUID[currentNode[i]];
        nextNodeBuffer[i] = pNextNode[currentNode[i]];
      }
    }
    #pragma unroll
    for (int i = 0; i<Stages; i++){
      int bid = bStart + i * TPB + tid;
      if (bid < numNew && !isDone[i] ){
        if (UUIDBuffer[i] == newUUID[i]){
          pLastVisitedNode[bid] = currentNode[i];
          pLastVisitedNodeStatus[bid] = 1;
          isDone[i] = true;
        } else {
          if (nextNodeBuffer[i] == NULL_NODE){
            pLastVisitedNode[bid] = currentNode[i];
            pLastVisitedNodeStatus[bid] = 0;
            isDone[i] = true;
          } else {
            currentNode[i] = nextNodeBuffer[i];
          }
        }
      }
    }
    bool isThreadDone;
    #pragma unroll
    for (int i = 0; i<Stages; i++){
      if (i == 0){
        isThreadDone = isDone[i];
      } else {
        isThreadDone = isThreadDone && isDone[i];
      }
    }

    if (isThreadDone){
      break;
    }
    counter ++;
  }
}


extern "C"
__global__ void hash_map_set_p2(
  const NodeType* __restrict__ pNewNode, //[numNewNodes]
  const NodeType* __restrict__ pLastVisitedNode, //[numNewNodes]
  NodeType* pNextNode,  //[numTotalNodes]
  int64_t* pInverse, //[numNewNodes]
  MutexType mutex //[numUniqueLastVisistedNode]
){
  constexpr int ThreadsPerBlock = _TPB_;
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  // lock 
  NodeType currentNode = pLastVisitedNode[bid];
  NodeType newNode = pNewNode[bid];
  MutexType currentMutex = &mutex[pInverse[bid]];

  mutex_lock_block(currentMutex);
  // NodeType currentNode = lastVisitedNode;

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
      // if (tid == 0){
      //   // printf("%d | %lld | %lld \n", bid, currentNode, nextNode);
      // }
      currentNode = nextNode;
    }
    counter ++;
  }
  mutex_unlock_block(currentMutex);
}
