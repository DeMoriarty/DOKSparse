template <typename T>
CUDA_DEVICE_INLINE
int binary_search_recursive(T *arr, int left, int right, T value)
{
  if (right >= left) {
    int mid = left + (right - left) / 2;

    // If the element is present at the middle
    // itself
    if (arr[mid] == value)
        return mid;

    // If element is smaller than mid, then
    // it can only be present in left subarray
    if (arr[mid] > value)
        return binary_search_recursive(arr, left, mid - 1, value);

    // Else the element can only be present
    // in right subarray
    return binary_search_recursive(arr, mid + 1, right, value);
  }

  // We reach here when element is not
  // present in array
    return -1;
}

template <typename T>
CUDA_DEVICE_INLINE
int binary_search_iterative(T *arr, int size, T value)
{   
    int left = 0;
    int right = size;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] == value)
            return mid;
        if (arr[mid] < value)
            left = mid + 1;
        else
            right = mid - 1;
    }
    return -1;
}

template <typename T> 
CUDA_DEVICE_INLINE
int binary_search_iterative_v2(T* arr, int size, T value)
{
  int low = 0;
  T v = value + 1;
  while (size > 0) {
    int half = size / 2;
    int other_half = size - half;
    int probe = low + half;
    int other_low = low + other_half;
    v = arr[probe];
    size = half;
    low = v < value ? other_low : low;
    if (v == value){
        return probe;
    }
  }
  return -1;
}

template <typename T> 
CUDA_DEVICE_INLINE
int binary_search_iterative_v3(T* arr, int __size, T value)
{
  int low = 0;
  int index = -1;
  #pragma unroll
  for (int size = __size; size > 0; size /= 2){
    int half = size / 2;
    int other_half = size - half;
    int probe = low + half;
    int other_low = low + other_half;
    T v = arr[probe];
    low = v < value ? other_low : low;
    // if (v == value){
    //     return probe;
    // }
    index = v == value ? probe : index;
    
  }
  return index;
}



template <typename T, int BatchSize, int Size> 
CUDA_DEVICE_INLINE
void binary_search_iterative_batched(
  T* arr, // [B, N] 
  T value,
  int indices[BatchSize]
) {
  #pragma unroll
  for (int d=0; d<BatchSize; d++){
    indices[d] = -1;
  }
  int low[BatchSize] = { 0 };
  #pragma unroll
  for (int size = Size; size > 0; size /= 2){
    int half = size / 2;
    int other_half = size - half;
    #pragma unroll
    for (int d=0; d<BatchSize; d++){
      int probe = low[d] + half;
      int other_low = low[d] + other_half;
      T v = arr[d * Size + probe];
      low[d] = v < value ? other_low : low[d];
      // if (v == value){
      //     return probe;
      // }
      indices[d]= v == value ? probe : indices[d];
    }
  } 
}
