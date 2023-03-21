template <
    typename T,
    int ndim
>
class StrictTensorAccessor{
    private:
        T* _dataPtr;
    public:
        ll_t _sizes[ndim];
        ll_t _strides[ndim];

        CUDA_DEVICE_INLINE
        StrictTensorAccessor(){
            
        };

        CUDA_DEVICE_INLINE
        StrictTensorAccessor(T* dataPtr,
                      ll_t sizes[ndim],
                      ll_t strides[ndim])
        {
            this->initialize(dataPtr, sizes, strides);
        }

        CUDA_DEVICE_INLINE
        StrictTensorAccessor(const ll_t* argPtr)
        {
            
            this->initialize(argPtr);
        }

        CUDA_DEVICE_INLINE
        void initialize(T* dataPtr,
                        ll_t *sizes,
                        ll_t *strides)
        {
            this->_dataPtr = dataPtr;
            this->set_sizes(sizes);
            this->set_strides(strides);
        }

        CUDA_DEVICE_INLINE
        void initialize(const ll_t* argPtr)
        {
            this->_dataPtr = reinterpret_cast<T*>(argPtr[0]);
            this->set_sizes(&argPtr[1]);
            this->set_strides(&argPtr[1 + ndim]);
        }

        CUDA_DEVICE_INLINE
        ll_t get_offset(ll_t indices[ndim]){
            ll_t offset = 0;
            #pragma unroll
            for (int i=0; i<ndim; i++){
                offset += indices[i] * this->_strides[i];
            }
            return offset;
        }

        CUDA_DEVICE_INLINE
        void get_offset(ll_t indices[ndim], ll_t &offset){
            offset = 0;
            #pragma unroll
            for (int i=0; i<ndim; i++){
                offset += indices[i] * this->_strides[i];
            }
        }

        CUDA_DEVICE_INLINE
        T get(ll_t indices[ndim]){
            ll_t offset = this->get_offset(indices);
            return this->_dataPtr[offset];
        }

        CUDA_DEVICE_INLINE
        void set(ll_t indices[ndim], T value){
            ll_t offset = this->get_offset(indices);
            this->_dataPtr[offset] = value;
        }

        CUDA_DEVICE_INLINE
        void get_index_from_offset(ll_t offset, ll_t indices[ndim]){
            #pragma unroll
            for (int i=0; i<ndim; i++){
                indices[i] = (offset / this->_strides[i]) % this->_sizes[i];
            }
        }

        // CUDA_DEVICE_INLINE
        // void set_strides(ll_t newStrides[ndim]){
        //     #pragma unroll
        //     for (int i=0; i<ndim; i++){
        //         this->_strides[i] = newStrides[i];
        //     }
        // }

        CUDA_DEVICE_INLINE
        void set_strides(const ll_t* newStrides){
            #pragma unroll
            for (int i=0; i<ndim; i++){
                this->_strides[i] = newStrides[i];
            }
        }

        // CUDA_DEVICE_INLINE
        // void set_sizes(ll_t newSizes[ndim]){
        //     #pragma unroll
        //     for (int i=0; i<ndim; i++){
        //         this->_sizes[i] = newSizes[i];
        //     }
        // }

        CUDA_DEVICE_INLINE
        void set_sizes(const ll_t* newSizes){
            #pragma unroll
            for (int i=0; i<ndim; i++){
                this->_sizes[i] = newSizes[i];
            }
        }

};

template <
    typename T
>
class TensorAccessor{
    private:
        T* _dataPtr;
    public:
        const ll_t *_sizes;
        const ll_t *_strides;
        ll_t _ndim;

        CUDA_DEVICE_INLINE
        TensorAccessor(){

        };

        CUDA_DEVICE_INLINE
        TensorAccessor(T* dataPtr,
                      ll_t *sizes,
                      ll_t *strides,
                      int ndim)
        {
            this->initialize(dataPtr, sizes, strides, ndim);
        }

        CUDA_DEVICE_INLINE
        TensorAccessor(const ll_t* argPtr, int ndim)
        {
            this->initialize(argPtr, ndim);
        }

        CUDA_DEVICE_INLINE
        void initialize(T* dataPtr,
                        ll_t *sizes,
                        ll_t *strides,
                        int ndim)
        {
            this->_ndim = ndim;
            this->_dataPtr = dataPtr;
            this->_sizes = sizes;
            this->_strides = strides;
        }

        CUDA_DEVICE_INLINE
        void initialize(const ll_t* argPtr, int ndim)
        {
            this->_ndim = ndim;
            this->_dataPtr = reinterpret_cast<T*>(argPtr[0]);
            this->_sizes = &argPtr[1];
            this->_strides = &argPtr[1 + ndim];
        }

        CUDA_DEVICE_INLINE
        ll_t get_offset(ll_t *indices){
            ll_t offset = 0;
            for (int i=0; i<this->_ndim; i++){
                offset += indices[i] * this->_strides[i];
            }
            return offset;
        }

        CUDA_DEVICE_INLINE
        void get_offset(ll_t *indices, ll_t &offset){
            offset = 0;
            for (int i=0; i<this->_ndim; i++){
                offset += indices[i] * this->_strides[i];
            }
        }

        CUDA_DEVICE_INLINE
        T get(ll_t *indices){
            ll_t offset = this->get_offset(indices);
            return this->_dataPtr[offset];
        }

        CUDA_DEVICE_INLINE
        void set(ll_t *indices, T value){
            ll_t offset = this->get_offset(indices);
            this->_dataPtr[offset] = value;
        }

        CUDA_DEVICE_INLINE
        void get_index_from_offset(ll_t offset, ll_t *indices){
            for (int i=0; i<this->_ndim; i++){
                indices[i] = (offset / this->_strides[i]) % this->_sizes[i];
            }
        }

        CUDA_DEVICE_INLINE
        void set_strides(const ll_t *newStrides){
            for (int i=0; i<this->_ndim; i++){
                this->_strides[i] = newStrides[i];
            }
        }

        CUDA_DEVICE_INLINE
        void set_sizes(const ll_t* newSizes){
            for (int i=0; i<this->_ndim; i++){
                this->_sizes[i] = newSizes[i];
            }
        }

};