/*
   Again as Fast as Possible
   1) just throwing in pragma fors
    benchmarks/1.txt                  0.020s  pass
    benchmarks/2a.txt                 0.323s  pass
    benchmarks/2b.txt                 0.347s  pass
    benchmarks/2c.txt                 0.331s  pass
    benchmarks/2d.txt                 0.420s  pass
    benchmarks/3.txt                 10.955s  pass
    benchmarks/4.txt                 43.928s  pass
*/


#include <stdlib.h>
#include <cmath>
#include <cstring>

typedef unsigned int u32;
typedef int s32;
typedef float f32;
typedef double f64;

#include <x86intrin.h>

// typedef __m256d f64x4;

typedef struct
{
    __m256d v;
} f64x4;

f64x4 operator+(f64x4 a, f64x4 b)
{
    f64x4 r;
    r.v = _mm256_add_pd(a.v, b.v);
    return(r);
}

f64x4 operator*(f64x4 a, f64x4 b)
{
    f64x4 r;
    r.v = _mm256_mul_pd(a.v, b.v);
    return(r);
}

f64 hadd(f64x4 a)
{
    f64 result = 0;
    f64 *Scalar = (f64 *)&a.v;
    for(u32 i = 0; i < 4; ++i)
    {
        result += Scalar[i];
    }
    return(result);
}

f64x4 loadu(f64 *a)
{
    f64x4 r;
    r.v = _mm256_loadu_pd(a);
    return(r);
}



void correlate(int ny, int nx, const float *data, float *result) 
{
    const s32 VecDim = 4;
    s32 VecCount = (nx + VecDim - 1) / VecDim;
    s32 PaddedX = VecDim * VecCount;

    const s32 BlockDim = 2;
    s32 BlockCountY = (ny + BlockDim - 1) / BlockDim;
    s32 PaddedY = BlockCountY * BlockDim; 

    f64 *NormData = (f64 *)malloc(PaddedY * PaddedX * sizeof(f64));

    #pragma omp parallel for
    for(s32 Row = 0; Row < ny; ++Row)
    {
        f64 Sum = 0;
        for(s32 Col = 0; Col < PaddedX; ++Col)
        {
            f64 val = (Col < nx) ? data[nx*Row + Col] : 0;
            NormData[PaddedX*Row + Col] = val;
            Sum += val;
        }

        f64 Mean = Sum/nx;
        f64 SumSq = 0;
        for(s32 Col = 0; Col < nx; ++Col)
        {
            f64 val = NormData[PaddedX*Row + Col] - Mean;
            NormData[PaddedX*Row + Col] = val;
            SumSq += val*val;
        }

        f64 InvStdY = 1/sqrt(SumSq);
        for(s32 Col = 0; Col < nx; ++Col)
        {
            NormData[PaddedX*Row + Col] *= InvStdY;
        }
    }

    #pragma omp parallel for
    for(s32 Row = ny; Row < PaddedY; ++Row)
    {
        for(s32 Col = 0; Col < PaddedX; ++Col)
        {
            NormData[PaddedX*Row + Col] = 0;
        }
    }


    // the output dim
    #pragma omp parallel for
    for(s32 Row = 0; Row < ny; ++Row)
    {
        for(s32 Col =  Row; Col < ny; ++Col)
        {
            f64x4 DotProds[BlockDim][BlockDim] = {};

            for(s32 VecIdx = 0; VecIdx < PaddedX; VecIdx += 4)
            {
                for(s32 i = 0; i < BlockDim; ++i)
                {
                    for(s32 j = 0; j < BlockDim; ++j)
                    {
                        if((Row + i < ny) && (Col + j < ny))
                        {
                            f64x4 x = loadu((f64*)(NormData + PaddedX*(Row + i) + VecIdx));
                            f64x4 y = loadu((f64*)(NormData + PaddedX*(Col + j) + VecIdx));
                            DotProds[i][j] = DotProds[i][j] + (x * y);
                        }
                    }
                }
               
            }
            

            for(s32 i = 0; i < BlockDim; ++i)
            {
                for(s32 j = 0; j < BlockDim; ++j)
                {
                    if((Row + i < ny) && (Col + j < ny))
                    {
                        result[ny*(Row + i) + (Col + j)] = hadd(DotProds[i][j]);
                    }
                }
            }
        }
    }
    free(NormData);
}
