#include <iostream>
#include <vector>
#include <cuda.h>
#include <algorithm>

#define CUDA_DEBUG

#ifdef CUDA_DEBUG

#define CUDA_CHECK_ERROR(err)           \
if (err != cudaSuccess) {          \
printf("Cuda error: %s\n", cudaGetErrorString(err));    \
printf("Error in file: %s, line: %i\n", __FILE__, __LINE__);  \
}                 \

#else

#define CUDA_CHECK_ERROR(err)

#endif

#define kNumBlockThreads 1024

// функция для CPU варианта
std::vector<unsigned int> find_perfect_numbers(unsigned int N)
{
    std::vector<unsigned int> out;
    unsigned int sum = 0;

    for (unsigned int i = 1; i<N; i++)
    {
        sum = 0;  // обнуление суммы делителей
        for (unsigned int n = 1; n <= i/2; n++)
        {
            if (i%n == 0)
                sum = sum + n;
        }
        if (i == sum) 
            out.push_back(i);
    }

    return out;
}

int cuda_blocks(int n_threads) {
    if((n_threads % kNumBlockThreads) == 0)
    {
        return (int)(n_threads/kNumBlockThreads);
    } else 
    {
        return (int)((n_threads/kNumBlockThreads) + 1);
    }
}

// функция для GPU варианта
__global__ void kernel_stretch(int *a, unsigned int N) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(i<N)
    {
        unsigned int sum = 0;
        for (unsigned int n = 1; n <= i/2; n++)
        {
            if (i%n == 0)
                sum = sum + n;
        }
        if (i == sum) 
            a[i] = sum;
        else
            a[i] = 0;
    }
}

// функция для GPU варианта с использованием shared типа памяти
__global__ void kernel_stretch_fast(int *a, unsigned int N) 
{
    __shared__ int temp[kNumBlockThreads];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(i<N)
    {
        unsigned int sum = 0;
        for (unsigned int n = 1; n <= i/2; n++)
        {
            if (i%n == 0)
                sum = sum + n;
        }
        if (i == sum) 
            temp[threadIdx.x] = sum;
        else
            temp[threadIdx.x] = 0;
    }

    if(i<N)
    {
        a[i] = temp[threadIdx.x];
    }
}


int main()
{
    
    unsigned int N;
    std::cout << "Input N: ";
    std::cin >> N;
    std::cout << std::endl;
    float timerValueGPU, timerValueCPU;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // --------------------------------------CPU----------------------------------------------
    std::cout << "--------------------------------------CPU----------------------------------------------\n";

    // старт таймера
    cudaEventRecord(start, 0);
    std::vector<unsigned int> all_perfect_numbers = find_perfect_numbers(N);
    // остановка таймера и вывод времени
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timerValueCPU,start, stop);

    std::cout << "CPU: " << "N = " << N << ". All time: " << timerValueCPU << " ms" << std::endl;

    for(int i = 0; i < all_perfect_numbers.size(); i++)
    {
        std::cout << "perfect number: " << all_perfect_numbers[i] << std::endl;
    }
    std::cout << std::endl;


    // --------------------------------------GPU----------------------------------------------
    std::cout << "--------------------------------------GPU----------------------------------------------\n";
    int *h_perfect_numbers;
    int *dev_perfect_numbers;
    h_perfect_numbers = new int[N];

    // старт таймера
    cudaEventRecord(start, 0);

    // выделяем память на GPU    
    CUDA_CHECK_ERROR(cudaMalloc((void**)&dev_perfect_numbers, sizeof(int)*N));
    
    // запускаем ядро
    kernel_stretch<<<cuda_blocks(N), kNumBlockThreads>>>(dev_perfect_numbers, N);

    // копируем память с GPU на HOST
    CUDA_CHECK_ERROR(cudaMemcpy(h_perfect_numbers,dev_perfect_numbers, sizeof(int)*N, cudaMemcpyDeviceToHost)); 
    
    // остановка таймера и вывод времени
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timerValueGPU,start, stop);
    std::cout << "GPU: " << "N = " << N << ". All time: " << timerValueGPU << " ms" << std::endl;

    for(int i = 0; i < N; i++)
    {
        if(h_perfect_numbers[i] != 0)
            std::cout << "perfect number: " << h_perfect_numbers[i] << std::endl;
    }
    std::cout << std::endl;

    cudaFree(dev_perfect_numbers);

    // старт таймера
    cudaEventRecord(start, 0);

    // выделяем память на GPU    
    CUDA_CHECK_ERROR(cudaMalloc((void**)&dev_perfect_numbers, sizeof(int)*N));

    // запускаем быстрое ядро
    kernel_stretch_fast<<<cuda_blocks(N), kNumBlockThreads>>>(dev_perfect_numbers, N);

    // копируем память с GPU на HOST
    CUDA_CHECK_ERROR(cudaMemcpy(h_perfect_numbers,dev_perfect_numbers, sizeof(int)*N, cudaMemcpyDeviceToHost)); 
    
    // остановка таймера и вывод времени
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timerValueGPU,start, stop);
    std::cout << "Fast GPU: " << "N = " << N << ". All time: " << timerValueGPU << " ms" << std::endl;

    for(int i = 0; i < N; i++)
    {
        if(h_perfect_numbers[i] != 0)
            std::cout << "perfect number: " << h_perfect_numbers[i] << std::endl;
    }
    std::cout << std::endl;

    delete[] h_perfect_numbers;
    cudaFree(dev_perfect_numbers);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}