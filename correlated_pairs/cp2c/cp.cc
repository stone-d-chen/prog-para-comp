/*
   VectorOps
   so removing the processing loop gets me down to 1.3 which is like 2.6 on the contest probably

*/


#include <stdlib.h>
#include <cmath>
#include <cstring>
#include <cstdio>

typedef unsigned int u32;
typedef unsigned long long u64;
typedef int s32;
typedef float f32;
typedef double f64;

#include <x86intrin.h>

// typedef __m256d f64x4;

typedef struct
{
    __m128 v;
} f32x4;
typedef struct
{
    __m256d v;
} f64x4;
typedef struct
{
    __m512d v;
} f64x8;


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
// f64x4 loadu(f64 *a)
// {
//     f64x4 r;
//     r.v = _mm256_loadu_pd(a);
//     return(r);
// }



f64x8 operator+(f64x8 a, f64x8 b)
{
    f64x8 r;
    r.v = _mm512_add_pd(a.v, b.v);
    return(r);
}

f64x8 operator*(f64x8 a, f64x8 b)
{
    f64x8 r;
    r.v = _mm512_mul_pd(a.v, b.v);
    return(r);
}

f64 hadd(f64x8 a)
{
    f64 result = 0;
    f64 *Scalar = (f64 *)&a.v;
    for(u32 i = 0; i < 8; ++i)
    {
        result += Scalar[i];
    }
    return(result);
}
f64x8 loadu(f64 *a)
{
    f64x8 r;
    r.v = _mm512_loadu_pd(a);
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
    const s32 VecDim = 8;
    s32 VecCount = (nx + VecDim - 1) / VecDim;
    s32 PaddedX = VecDim * VecCount;

    u64 BeginProc = __rdtsc();
    f64 *NormData = (f64 *)malloc(ny * PaddedX * sizeof(f64));

    for(s32 Row = 0; Row < ny; ++Row)
    {
        for(s32 Col = 0; Col < nx; ++Col)
        {
            NormData[PaddedX*Row + Col] = data[nx*Row + Col];
        }
        for(s32 Col = nx; Col < PaddedX; ++Col)
        {
            NormData[PaddedX*Row + Col] = 0;
        }
        f64 Sum = 0;
        for(s32 Col = 0; Col < PaddedX; ++Col)
        {
            Sum += NormData[PaddedX*Row + Col];
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
    u64 EndProc = __rdtsc();

    // the output dim
    for(s32 Row = 0; Row < ny; ++Row)
    {
        for(s32 Col =  Row; Col < ny; ++Col)
        {
            f64x8 DotProds = {};

            for(s32 VecIdx = 0; VecIdx < PaddedX; VecIdx += VecDim)
            {
                f64x8 x = loadu((f64*)(NormData + PaddedX*Row + VecIdx));
                f64x8 y = loadu((f64*)(NormData + PaddedX*Col + VecIdx));
                DotProds = DotProds + (x * y);
            }
            f64 FinalSum = hadd(DotProds);
            result[ny*Row + Col] = FinalSum;
        }
    }
    u64 EndComp = __rdtsc();

    u64 CompTime = EndComp - EndProc;
    u64 TotalTime = EndComp - BeginProc;

    printf("Percent Time Compute: %f", (f32) CompTime * 100 / TotalTime);

    free(NormData);

}
