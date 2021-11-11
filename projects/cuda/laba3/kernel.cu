#include <stdio.h>
#include <ctime>
#include <iostream>
#include <set>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

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

#define E 1e-3


int cuda_blocks(int n_threads) {
    if((n_threads % kNumBlockThreads) == 0)
    {
        return (int)(n_threads/kNumBlockThreads);
    } else 
    {
        return (int)((n_threads/kNumBlockThreads) + 1);
    }
}


__device__ __host__ float func(float x){
    return tanf(1.262*x) - 1.84 * x;
}


__device__ __host__ float func_derivative(float x){
    return 1.262 / (cosf(1.262*x)*cosf(1.262*x)) - 1.84;
}


__device__ __host__ float func_derivative_second(float x){
    return 3.1853 * (sinf(1.262*x) / powf(cosf(1.262*x), 3));
}


__global__ void newton_method(float *c, double step, float A, int N){   
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i+1<N)
    {
        //Считаем границы для gpu
        float a = A + (float)i * step;
        float b = A + (float)(i+1) * step;
        // printf("\ni: %i;a: %.5f ;b: %.5f\n\n", i, a, b);
        if (func(a) * func(b) * 1.0 > 0)
            return;

        if(!isfinite(func(a)))
            return;
        
        if(!isfinite(func(b)))
            return;    
        double x;

        if (func(a) * func_derivative_second(a) > 0)
            x = a;
        else
            x = b;
        do
        {
            x = x - func(x) / func_derivative(x);
            if(!isfinite(x) || !isfinite(func(x)) || !isfinite(func_derivative(x)))
                return;

        } while (fabs(func(x)) >= E);  // цикл ищет корень пока его значение больше заданой точности

        c[i] = x;
    }
}


void cpu(){
    int start = clock(), time;
    double f=1, df;
    double c=0.5; 
    int n=0;
    while (fabs(f)>=E && n < 20000)
    {
        f = func(c);
        df = func_derivative(c);
        c = c - f/df * 0.1; // уменьшаем шаг для того, чтобы не происходило "переобучения". Если не умножать на 0.01, то корень уйдёт в бесконечность (мог бы проверять на финитность и уменьшать шаг, но для данной лабораторной это слишком)
        n++;
    }
    time = clock() - start;
 	printf("==============================   CPU TIME   ===============================\n");
    printf("Equation root = %lf\n",c);
    printf("Iteration number: n = %d\n",n); 
 	printf("\nCPU compute time: %.5f microseconds\n\n", time*1000);
}

void gpu(){
    size_t free, total;
    printf("\n");
    cudaMemGetInfo(&free, &total);
    //printf("%zu KB free of total %zu KB\n", free / 1024, total / 1024);

    float B = 10., A = -10.; // common borders

    cudaEvent_t start, stop;
	float gpuTime = 0.0f;
	CUDA_CHECK_ERROR(cudaEventCreate(&start));
	CUDA_CHECK_ERROR(cudaEventCreate(&stop));
    CUDA_CHECK_ERROR(cudaEventRecord(start, 0));

    const int N = free * 0.002 / sizeof(float);
    const int BLOCKS = cuda_blocks(N);
    float step = fabs(A - B) / N;

    thrust::host_vector<float> c (N);
    thrust::device_vector<float> dev_c (N);

    newton_method<<<BLOCKS, kNumBlockThreads, 0, 0>>>(thrust::raw_pointer_cast(dev_c.data()), step, A, N);

    thrust::copy(dev_c.begin(), dev_c.end(), c.begin());
    thrust::sort(c.begin(), c.end());
    thrust::unique(thrust::host, c.begin(), c.end());

    CUDA_CHECK_ERROR(cudaEventRecord(stop, 0));
	CUDA_CHECK_ERROR(cudaEventSynchronize(stop));
  	CUDA_CHECK_ERROR(cudaEventElapsedTime(&gpuTime, start, stop));
    printf("==============================   GPU TIME   ===============================\n");
 	printf("\nGPU compute time: %.5f microseconds\n\n", gpuTime);


    std::set<float> out;
    for(int i = 0;i< c.size(); i++)
    {
        out.insert(round(c[i]*10000)/10000); // округляем до x.xxxxx
    }
    for(auto &i: out)
    {
        printf("ROOT: %.5f \n", i);
    }
}

int main(void){
    gpu();
    cpu();
    return 0;
}