void step(float* r, const float* d, int n) {
  for(int i = 0; i < n; ++i)
  {
    for(int j = 0; j < n; ++j)
    {
        float v = 1e100;
        for(int k = 0; k < n; ++k)
        {
            float x = d[n*i + k];
            float y = d[n*k + j];
            v = min(v, x+ y);
        }

        r[n*i + j] = v;
    }
  }
  
}

/*
    n x n x n units of work;
    we'll let 1 thread do 1 x 1 x n of work
    We'll create dimBlocks(16, 16);

    we'll create dimGrid(n/16, n/16);

*/



__global__
void mykernel(float *r, float *d, int n)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    if(i >= n || j >= n) return;

    int v = HUGE_VALF;

    for(k = 0; k < n; ++k)
    {
        float x = d[n*i + k];
        float y = d[n*k + j];
        v = min(v, x + y);
    }
    r[n*i + j] = v;
}

void step(float *r, const float *d, int n)
{
    float *dGPU = NULL;
    cudaMalloc((void **)&dGPU, sizeof(float) * n * n );

    float *rGPU = NULL;
    cudaMalloc((void **) &rGPU, sizeof(float) * n * n);

    cudaMemcpy(dGPU, d, sizeof(float) * n * n, cudaMemcpyHostToDevice);


    dim3 dimBlock(16, 16);
    dim3 dimGrid(divup(n, dimBlock.x), divup(n, dimBlock.y));

    mykernel<<<dimGrid, dimBlock>>>(dGPU, rGPU, n);

    cudaMemcpy(r, rGPU, sizeof(float) * n * n, cudaMemcpyDeviceToHost);




}