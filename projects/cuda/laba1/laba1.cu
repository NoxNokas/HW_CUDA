#include <stdio.h>
#include <iostream>


#define kNumBlockThreads 512

template<typename T>
__global__ void kernel_stretch(T *a, unsigned int n) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x) {
      a[idx] = a[idx] * a[idx];
  }
}

template<typename T>
void cpu_foo(T *a, unsigned int n) {
    for(unsigned int i = 0; i < n; i++) {
        a[i] = a[i] * a[i];
    }
}

int cuda_blocks(int n_threads) {
    return (n_threads + kNumBlockThreads - 1) / kNumBlockThreads;
}

int main( void )
{
    unsigned int start_time;
    unsigned int end_time;
    unsigned int search_time;
    
    const unsigned int MAX_ARRAY = 1000000;
    for(unsigned int N = 10; N<MAX_ARRAY; N = N * 10){
        

        int a[N];
        int *dev_a;

        // инициализируем массив
        for (int i = 0; i < N; i++){
            a[i] = i;
        }
        std::cout << std::endl;
        
        // выделяем память на GPU
        cudaMalloc((void**)&dev_a, N*sizeof(int));
        // копируем память с HOST на GPU
        cudaMemcpy(dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice);
        start_time =  clock(); // начальное время
        // запускаем ядро
        kernel_stretch<<<cuda_blocks(N), 512>>>(dev_a, N);
        end_time = clock(); // конечное время
        search_time = end_time - start_time; // искомое время
        // копируем память с GPU на HOST
        cudaMemcpy(a, dev_a, N*sizeof(int), cudaMemcpyDeviceToHost);
        // освобождаем память
        cudaFree(dev_a);
        
        std::cout << "GPU; " << "N = " << N << ". All time: " << search_time;
        // for (int i = 0; i < N; i++){
        //     printf("%f, ", a[i]);
        // }
        std::cout << std::endl;
    }

    // код для CPU
    for(unsigned int N = 10; N<MAX_ARRAY; N = N * 10){
        

        int a[N];

        // инициализируем массив
        for (int i = 0; i < N; i++){
            a[i] = i;
        }
        std::cout << std::endl;
        start_time =  clock(); // начальное время
        cpu_foo(a, N);

        end_time = clock(); // конечное время
        search_time = end_time - start_time; // искомое время
        std::cout << "CPU; " << "N = " << N << ". All time: " << search_time;
    }

    std::cout << std::endl;
    return 0;
}