/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/

#include <cuda_runtime.h>

typedef float f32;

__global__ void kernel(int ny, int nx, const float *data, float *result)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;

    if(i >= ny || j >= ny) return;

    f32 v = 0;
    for(int k = 0; k < nx; ++k)
    {
        v += data[nx*i + k] * data[nx * j + k];
    }

    result[ny * i + j] = v;
}

int divup(int n, int factor) {
    int Result = (n + factor - 1) / factor;
    return(Result);
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
            NormData[nx*y + x] = val;
            sum += val;
        }
        int mean = sum/nx;

        f32 SumSq = 0;
        for(int x = 0; x < nx; ++x)
        {
            NormData[nx*y + x] -= mean;
            SumSq += NormData[nx*y + x] * NormData[nx*y + x];
        }

        f32 InvStd = 1/sqrtf(SumSq);

        for(int x = 0; x < nx; ++x)
        {
            NormData[nx*y + x] *= InvStd;
        }
    }

    f32 Result = 0;
    for(int j = 0; j < ny; ++j)
    {
        for(int i = j; i < ny; ++i)
        {
            for(int k = 0; k < nx; ++k)
            {
                f32 x = NormData[nx*i + k];
                f32 y = NormData[nx*j + k];
                Result += x * y;
            }
            result[ny*j + i] = Result;
        }
    }

    dim3 dimBlock(16,16);
    dim3 dimGrid(divup(nx, 16), divup(ny, 16));

    f32 *dataGPU;
    f32 *resultGPU;

   cudaMalloc((void**)&dataGPU, sizeof(f32) * ny * nx);
   cudaMalloc((void**)&resultGPU, sizeof(f32) * ny * ny);

   cudaMemcpy(dataGPU, data, sizeof(f32) * ny * nx, cudaMemcpyHostToDevice);

   kernel <<< dimGrid, dimBlock >>>(nx, ny, dataGPU, resultGPU);

   cudaDeviceSynchronize();

   cudaMemcpy(result, resultGPU, sizeof(f32) * ny * ny, cudaMemcpyDeviceToHost);

   cudaFree(dataGPU);
   cudaFree(resultGPU);

}
