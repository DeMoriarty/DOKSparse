template <typename T, int StackCap>
class Stack{
  private:
    int _stackSize = 0;
    T _stack[StackCap];

  public:
    CUDA_DEVICE_INLINE
    Stack(){
    }

    CUDA_DEVICE_INLINE 
    bool is_empty(){
      return _stackSize <= 0;
    }

    CUDA_DEVICE_INLINE
    bool is_full(){
      return _stackSize >= StackCap - 1;
    }

    CUDA_DEVICE_INLINE
    int size(){
      return _stackSize;
    }

    CUDA_DEVICE_INLINE
    int capacity(){
      return StackCap;
    }

    CUDA_DEVICE_INLINE
    void fill(T item){
      #pragma unroll
      for (int i=0; i < StackCap; i++){
        _stack[i] = item;
      }
    }

    CUDA_DEVICE_INLINE
    void push(T item){
      if (is_full()){
        return;
      } else {
        #pragma unroll
        for (int i = StackCap - 1; i >= 1; i--){
          _stack[i] = _stack[i - 1];
        }
        _stack[0] = item;
        _stackSize ++;
      }
    }

    CUDA_DEVICE_INLINE
    void pop(T &out){
      if (is_empty()){
        return;
      } else {
        out = _stack[0];
        #pragma unroll
        for (int i=0; i<StackCap-1; i++){
          _stack[i] = _stack[i+1];
        }
        _stackSize--;
      }
    }

    CUDA_DEVICE_INLINE
    T pop(){
      T outItem;
      if (!is_empty()) {
        outItem = _stack[0];
        #pragma unroll
        for (int i=0; i<StackCap-1; i++){
          _stack[i] = _stack[i+1];
        }
        _stackSize--;
      }
      return outItem;
    }

};
