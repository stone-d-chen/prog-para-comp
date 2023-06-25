/*
   VectorOps
*/


#include <stdlib.h>
#include <cmath>

typedef unsigned int u32;
typedef int s32;
typedef float f32;
typedef double f64;

#include <x86intrin.h>

typedef __m256d f64x4;

typedef struct
{
    __mm256d v;
};


f64x4 operator+(f64x4 a, f64x4 b)
{
    f64x4 r = _mm256_add_pd(a, b);
    return(r);
}

f64x4 operator*(f64x4 a, f64x4 b)
{
    f64x4 r = _mm256_mul_pd(a, b);
    return(r);
}

f64 hadd(f64x4 a)
{
    f64 result;
    f64 *Scalar = (f64 *)&a;
    for(u32 i = 0; i < 4; ++i)
    {
        result += Scalar[i];
    }
    return(result);

}


void correlate(int ny, int nx, const float *data, float *result) 
{
    const s32 VecDim = 4;
    s32 VecCount = (nx + VecDim - 1) / VecDim;
    s32 PaddedX = VecDim * VecCount;

    f64 *NormData = (f64 *)malloc(ny * PaddedX * sizeof(f64));

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

    f64x4 *VecNormData = (f64x4 *)NormData;


    // the output dim
    for(s32 Row = 0; Row < ny; ++Row)
    {
        for(s32 Col =  Row; Col < ny; ++Col)
        {
            f64x4 DotProds = {};
            for(s32 VecIdx = 0; VecIdx < VecCount; ++VecIdx)
            {
                f64x4 x = VecNormData[PaddedX*Row + VecIdx];
                f64x4 y = VecNormData[PaddedX*Col + VecIdx];
                DotProds = DotProds + (x * y);
            }
            f64 FinalSum = hadd(DotProds);
            result[ny*Row + Col] = FinalSum;
        }
    }
    free(NormData);
}
