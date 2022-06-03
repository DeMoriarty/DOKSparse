#define BoolType uint8_t
#define MaxIterLimit 100000
#define NULL_NODE -1

template <int Ndim, typename KeyType>
class Hashmap {
  private:
    int64_t _alpha1[Ndim];
    int64_t _alpha2[Ndim];
    int64_t _beta1[Ndim];
    int64_t _beta2[Ndim];
    int64_t _prime1[Ndim];
    int64_t _prime2[Ndim];
    const int64_t *_pHeadNode;
    const int64_t *_pNextNode;
    const int64_t *_pUUID;
    int _numBuckets;

  public:
    CUDA_DEVICE_INLINE
    Hashmap(const int64_t* pAlpha1,
            const int64_t* pAlpha2,
            const int64_t* pBeta1,
            const int64_t* pBeta2,
            const int64_t* pPrime1,
            const int64_t* pPrime2,
            const int64_t* pHeadNode,
            const int64_t* pNextNode,
            const int64_t* pUUID,
            int numBuckets
            )
            : _pHeadNode(pHeadNode)
            , _pNextNode(pNextNode)
            , _pUUID(pUUID)
            , _numBuckets(numBuckets)
    {
      #pragma unroll
      for (int i=0; i<Ndim; i++){
        _alpha1[i] = pAlpha1[i];
        _alpha2[i] = pAlpha2[i];
        _beta1[i] = pBeta1[i];
        _beta2[i] = pBeta2[i];
        _prime1[i] = pPrime1[i];
        _prime2[i] = pPrime2[i];
      }
    }

    CUDA_DEVICE_INLINE
    int64_t get_hash(KeyType key[Ndim]){
      int64_t hash_code = (key[0] * _alpha1[0] + _beta1[0]) % _prime1[0];
      #pragma unroll
      for (int i=1; i<Ndim; i++){
        hash_code *= (key[i] * _alpha1[i] + _beta1[i]) % _prime1[i];
      }
      hash_code = llabs(hash_code);
      hash_code = hash_code % _numBuckets;
      return hash_code;
    }

    template <int NSubkeyDim>
    CUDA_DEVICE_INLINE
    int64_t get_hash_subkey(KeyType key[NSubkeyDim], int subkeyInds[NSubkeyDim]){
      int64_t hash_code = (key[subkeyInds[0]] * _alpha1[subkeyInds[0]] + _beta1[subkeyInds[0]]) % _prime1[subkeyInds[0]];
      #pragma unroll
      for (int i=1; i<NSubkeyDim; i++){
        hash_code *= (key[subkeyInds[i]] * _alpha1[subkeyInds[i]] + _beta1[subkeyInds[i]]) % _prime1[subkeyInds[i]];
      }

      hash_code = llabs(hash_code);
      hash_code = hash_code % _numBuckets;
      return hash_code;
    }

    CUDA_DEVICE_INLINE
    int64_t get_uuid(KeyType key[Ndim]){
      int64_t uuid = (key[0] * _alpha2[0] + _beta2[0]) % _prime2[0];
      #pragma unroll
      for (int i=1; i<Ndim; i++){
        uuid *= (key[i] * _alpha2[i] + _beta2[i]) % _prime2[i];
      }
      uuid = llabs(uuid);
      return uuid;
    }

    CUDA_DEVICE_INLINE
    int64_t find_bucket_head(int64_t hashCode){
      int64_t bucketHead = _pHeadNode[hashCode];
      return bucketHead;
    }

    CUDA_DEVICE_INLINE
    void find_node(KeyType key[Ndim], int64_t &lastVisitedNode, bool &isKeyFound){
      int64_t hashCode = get_hash(key);
      int64_t UUID = get_uuid(key);
      int64_t nextNode = find_bucket_head(hashCode);
      if (nextNode == NULL_NODE){
        isKeyFound = false;
        return;
      }

      int64_t currentNode = nextNode;
      int counter = 0;
      while (counter < MaxIterLimit){
        int64_t currentUUID = _pUUID[currentNode];
        nextNode = _pNextNode[currentNode];
        if (currentUUID == UUID){
          lastVisitedNode = currentNode;
          isKeyFound = true;
          return;

        } else if (nextNode == NULL_NODE) {
          isKeyFound = false;
          return;
        } else {
          currentNode = nextNode;
        }
        counter++;
      }
      isKeyFound = false;
    } 

    CUDA_DEVICE_INLINE
    void find_node(int64_t hashCode, int64_t UUID, int64_t &lastVisitedNode, bool &isKeyFound){
      int64_t nextNode = _pHeadNode[hashCode];
      if (nextNode == NULL_NODE){
        isKeyFound = false;
        return;
      }
      int64_t currentNode = nextNode;
      int counter = 0;
      while (counter < MaxIterLimit){
        int64_t currentUUID = _pUUID[currentNode];
        nextNode = _pNextNode[currentNode];
        if (currentUUID == UUID){
          lastVisitedNode = currentNode;
          isKeyFound = true;
          return;

        } else if (nextNode == NULL_NODE) {
          isKeyFound = false;
          return;
        } else {
          currentNode = nextNode;
        }
        counter++;
      }
      isKeyFound = false;
    } 

    template <int Stages>
    CUDA_DEVICE_INLINE
    void find_nodes(int64_t hashCode[Stages], int64_t UUID[Stages], int64_t lastVisitedNode[Stages], bool isKeyFound[Stages], int nValidStages){
      int64_t nextNode[Stages];
      int64_t currentNode[Stages];
      bool isDone[Stages] = {false};

      #pragma unroll
      for (int i=0; i<Stages; i++){
        isKeyFound[i] = false;

        if (i < nValidStages){
          nextNode[i] = _pHeadNode[hashCode[i]];
          if (nextNode[i] == NULL_NODE){
            isDone[i] = true;
            lastVisitedNode[i] = -1;
          } else {
            currentNode[i] = nextNode[i];
          }
        } else {
          isDone[i] = true;
        }
      }

      int counter = 0;
      while (counter < MaxIterLimit){
        int64_t currentUUID[Stages];
        #pragma unroll
        for (int i=0; i<Stages; i++){
          if (!isDone[i]){
            currentUUID[i] = _pUUID[currentNode[i]];
            nextNode[i] = _pNextNode[currentNode[i]];
          }
        }

        #pragma unroll
        for (int i=0; i<Stages; i++){
          if (!isDone[i]){
            lastVisitedNode[i] = currentNode[i];
            if (currentUUID[i] == UUID[i]){
              isDone[i] = true;
              isKeyFound[i] = true;
            } else if (nextNode[i] == NULL_NODE) {
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

        counter++;
      }
    } 

    template <int Stages>
    CUDA_DEVICE_INLINE
    void find_nodes(KeyType keys[Stages][Ndim], int64_t lastVisitedNode[Stages], bool isKeyFound[Stages], int nValidStages){
      int64_t hashCode[Stages];
      int64_t UUID[Stages];
      #pragma unroll
      for (int i=0; i<Stages; i++){
        if (i < nValidStages){
          hashCode[i] = get_hash(keys[i]);
          UUID[i] = get_uuid(keys[i]);
        }
      }
      find_nodes<Stages>(hashCode, UUID, lastVisitedNode, isKeyFound, nValidStages);
    }
};