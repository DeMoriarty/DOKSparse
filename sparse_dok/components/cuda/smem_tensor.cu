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