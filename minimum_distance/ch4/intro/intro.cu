__device__ void foo(int i, int j) {};

__global__ void mykernel()
{
    int i = blockIdx.x;
    int j = threadIdx.x;

    foo(i,j);
}

int main()
{
    mykernel <<< 100, 128 >>> ();
    cudaDeviceSynchronize();
}

/*
    for(int i = 0; i < 100; ++i)
    {
        for(int j = 0; j < 128; ++j)
        {
            foo(i,j);
        }
    }


*/


/*
    Multidimensional grids and blocks
    Often it is convenient to index blocks and threads with 2-dimensional or 3-dimensional indexes.
    Here is an example in which we create 20 × 5 blocks, each with 16 × 8 threads:
*/


__device__
void foo(int ix, int iy, int jx, int jy) {}

__global__
void mykernel()
{
    int ix = blockIdx.x;
    int iy = blockIdx.y;

    int jx = threadIdx.x;
    int jy = threadIdx.y;

    foo(ix, iy, jx, jy);
}

int main()
{
            //  x   y
    dim3 dimGrid(20, 5);
    dim3 dimBlock(16, 8);

    mykernel<<<dimGrid, dimBlock>>> ();
    cudaDeviceSynchronize();

    /*
        for(int iy = 0; iy < blockIdx.y; ++iy) 
            for(int ix = 0; ix < blockIdx.x; ++ix)
                for(int jy = 0; jy < threadIdx.y; ++iy)
                    for(int jx = 0; jx < threadIdx.x; ++ix)
                        foo(ix, iy, jx, jy);
    */
}

