/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/

#include <iostream>
#include <cuda_runtime.h>





static inline void check(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << context << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK(x) check(x, #x)

typedef float f32;
typedef double f64;

int divup(int n, int factor) {
    int Result = (n + factor - 1) / factor;
    return(Result);
}

__global__ void kernel(int ny, int nx, const float *data, float *result)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;

    if(i >= ny || j >= ny) return;

    float v = 0;
    for(int k = 0; k < nx; ++k)
    {
        v += data[nx*i + k] * data[nx * j + k];
    }

    result[ny * j + i] = v;
}


void correlate(int ny, int nx, const float *data, float *result)
{
    f32 *NormData = (f32 *)malloc(ny*nx*sizeof(f32));


    for(int y = 0; y < ny; ++y)
    {
        f32 sum = 0;
        for(int x = 0; x < nx; ++x)
        {
            f32 val = data[nx * y + x];
            sum += val;
            NormData[nx*y + x] = val;
        }
        f32 mean = sum/nx;

            printf("mean = %f\n", mean);


        f32 SumSq = 0;
        for(int x = 0; x < nx; ++x)
        {
            f32 val = NormData[nx*y + x] - mean;
            NormData[nx*y + x] = val;
            printf("%f\n", val);

            SumSq += val * val;
        }

        f32 InvStd = 1/sqrt(SumSq);

        for(int x = 0; x < nx; ++x)
        {
            NormData[nx*y + x] *= InvStd;
        }
    }


    dim3 dimBlock(8,8);
    dim3 dimGrid(divup(ny, 8), divup(ny, 8));

    f32 *dataGPU;
    f32 *resultGPU;

   cudaMalloc((void**)&dataGPU, sizeof(f32) * ny * nx);
   cudaMalloc((void**)&resultGPU, sizeof(f32) * ny * ny);

   cudaMemcpy(dataGPU, NormData, sizeof(f32) * ny * nx, cudaMemcpyHostToDevice);

   kernel <<< dimGrid, dimBlock >>>(ny, nx, dataGPU, resultGPU);
   cudaGetLastError();

   cudaDeviceSynchronize();

   cudaMemcpy(result, resultGPU, sizeof(f32) * ny * ny, cudaMemcpyDeviceToHost);
   cudaFree(dataGPU);
   cudaFree(resultGPU);
   cudaGetLastError();



}

const int ny = 7;
const int nx = 7;
const float test[] = {83, 88.5, 88.2, 89.5, 96.2, 98.1, 99, 100, 101.2, 
104.6, 108.4, 110.8, 112.6, 114.2, 115.7, 116.9, 234.289, 259.426, 
258.054, 284.599, 328.975, 346.999, 365.385, 363.112, 397.469, 
419.18, 442.769, 444.546, 482.704, 502.601, 518.173, 554.894, 
235.6, 232.5, 368.2, 335.1, 209.9, 193.2, 187, 357.8, 290.4, 
282.2, 293.6, 468.1, 381.3, 393.1, 480.6, 400.7, 159, 145.6, 
161.6, 165, 309.9, 359.4, 354.7, 335, 304.8, 285.7, 279.8, 263.7, 
255.2, 251.4, 257.2, 282.7, 107.608, 108.632, 109.773, 110.929, 
112.075, 113.27, 115.094, 116.219, 117.388, 118.734, 120.445, 
121.95, 123.366, 125.368, 127.852, 130.081, 1947, 1948, 1949, 
1950, 1951, 1952, 1953, 1954, 1955, 1956, 1957, 1958, 1959, 1960, 
1961, 1962, 60.323, 61.122, 60.171, 61.187, 63.221, 63.639, 64.989, 
63.761, 66.019, 67.857, 68.169, 66.513, 68.655, 69.564, 69.331, 
70.551};
float res[ny * ny];

int main()
{

    correlate(ny, nx, test,res);

    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < nx; ++j) {
            std::cout << res[i*nx + j] << " ";
        }
        std::cout << "\n";
    }
}