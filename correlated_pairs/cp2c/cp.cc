/*
   VectorOps
   so removing the processing loop gets me down to 1.3 which is like 2.6 on the contest probably

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
typedef struct
{
    __m128 v;
} f32x4;


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
f64x4 load_F32(const f32 *a)
{
    __m128 val = _mm_loadu_ps(a);
    f64x4 r;
    r.v = _mm256_cvtps_pd(val);
    return(r);
}
void storeu(f64 *a, f64x4 b)
{
    _mm256_storeu_pd(a, b.v);
}


void correlate(int ny, int nx, const float *data, float *result) 
{
    const s32 VecDim = 4;
    s32 VecCount = (nx + VecDim - 1) / VecDim;
    s32 PaddedX = VecDim * VecCount;

    f64 *NormData = (f64 *)malloc(ny * PaddedX * sizeof(f64));
    memset(NormData, 0, ny * PaddedX * sizeof(f64));

    // for(s32 Row = 0; Row < ny; ++Row)
    // {
    //     f64 Sum = 0;
    //     for(s32 Col = 0; Col < PaddedX; ++Col)
    //     {
    //         f64 val = (Col < nx) ? data[nx*Row + Col] : 0;
    //         NormData[PaddedX*Row + Col] = val;
    //         Sum += val;
    //     }

    //     f64 Mean = Sum/nx;
    //     f64 SumSq = 0;
    //     for(s32 Col = 0; Col < nx; ++Col)
    //     {
    //         f64 val = NormData[PaddedX*Row + Col] - Mean;
    //         NormData[PaddedX*Row + Col] = val;
    //         SumSq += val*val;
    //     }

    //     f64 InvStdY = 1/sqrt(SumSq);
    //     for(s32 Col = 0; Col < nx; ++Col)
    //     {
    //         NormData[PaddedX*Row + Col] *= InvStdY;
    //     }
    // }

    for(s32 Row = 0; Row < ny; ++Row)
    {
        f64x4 Sum = {};
        for(s32 Col = 0; Col < nx - 4; Col+=4)
        {
            f64x4 val = load_F32(data + nx*Row + Col);
            Sum = Sum + val;
            storeu(NormData + PaddedX*Row + Col, val);
        }

        f64 Mean = hadd(Sum)/nx;
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

    // the output dim
    for(s32 Row = 0; Row < ny; ++Row)
    {
        for(s32 Col =  Row; Col < ny; ++Col)
        {
            f64x4 DotProds = {};

            for(s32 VecIdx = 0; VecIdx < PaddedX; VecIdx += 4)
            {
                f64x4 x = loadu((f64*)(NormData + PaddedX*Row + VecIdx));
                f64x4 y = loadu((f64*)(NormData + PaddedX*Col + VecIdx));
                DotProds = DotProds + (x * y);
            }
            f64 FinalSum = hadd(DotProds);
            result[ny*Row + Col] = FinalSum;
        }
    }
    free(NormData);
}

#if 0
const int nx = 101;
const int ny = 93;
const float d[nx*ny] = {};
float r[ny*ny];


int main()
{
    correlate(ny, nx, d, r);
}

#endif