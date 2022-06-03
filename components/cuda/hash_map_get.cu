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


extern "C"
__global__ void hash_map_get(
  const NodeType* __restrict__ pHashCode,
  const NodeType* __restrict__ pQueryUUID,
  const NodeType* __restrict__ pUUID,
  const NodeType* __restrict__ pNextNode,
  const NodeType* __restrict__ pHeadNode,
  NodeType* pIndices,
  BoolType* pIsKeyFound,
  int nQuery
){
  constexpr int TPB = _TPB_;
  constexpr int Stages = _STAGES_;
  constexpr int NewPairPerBlock = TPB * Stages;

  const int tid = threadIdx.x; 
  const int bStart = blockIdx.x * NewPairPerBlock;

  NodeType currentNode[Stages];
  NodeType hashCode[Stages];
  NodeType queryUUID[Stages];
  NodeType nextNode[Stages];
  NodeType currentUUID[Stages];
  bool isDone[Stages] = { false };

  #pragma unroll
  for (int i=0; i<Stages; i++){
    int bid = bStart + i * TPB + tid;
    if (bid < nQuery){
      hashCode[i] = pHashCode[bid];
      queryUUID[i] = pQueryUUID[bid];
      nextNode[i] = pHeadNode[hashCode[i]];
      if (nextNode[i] == NULL_NODE){
        // pIndices[bid] = NULL_NODE;
        isDone[i] = true;
      } else {
        currentNode[i] = nextNode[i];
      }
    } else {
      isDone[i] = true;
    }
  }

  // #pragma unroll
  // for (int i=0; i<Stages; i++){
  //   int bid = bStart + i * TPB + tid;
  //   if (!isDone[i]){
      
  //   }
  // }

  int counter = 0;
  while (counter < MaxIterLimit){
    // NodeType nextNode[Stages];
    #pragma unroll
    for (int i = 0; i<Stages; i++){
      if (!isDone[i]){
        currentUUID[i] = pUUID[currentNode[i]];
        nextNode[i] = pNextNode[currentNode[i]];
      }
    }

    #pragma unroll
    for (int i = 0; i<Stages; i++){
      int bid = bStart + i * TPB + tid;
      if (!isDone[i]){
        if (currentUUID[i] == queryUUID[i]){
          pIndices[bid] = currentNode[i];
          pIsKeyFound[bid] = (BoolType) true;
          isDone[i] = true;

        } else if ( nextNode[i] == NULL_NODE ) {
          // pIndices[bid] = currentNode[i];
          isDone[i] = true;

        } else {
          currentNode[i] = nextNode[i];
        }
      }
    }

    bool isThreadDone = isDone[0];
    #pragma unroll
    for (int i = 1; i<Stages; i++){
      isThreadDone = isThreadDone && isDone[i];
    }

    if (isThreadDone){
      break;
    }
    counter ++;
  }
}