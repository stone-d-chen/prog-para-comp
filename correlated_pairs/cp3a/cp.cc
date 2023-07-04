#include <stdlib.h>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <stdio.h>

typedef unsigned int u32;
typedef int s32;
typedef float f32;
typedef double f64;
typedef unsigned long long u64;

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


inline f64x4 operator+(f64x4 a, f64x4 b)
{
    f64x4 r;
    r.v = _mm256_add_pd(a.v, b.v);
    return(r);
}
inline f64x4 operator-(f64x4 a, f64x4 b)
{
    f64x4 r;
    r.v = _mm256_sub_pd(a.v, b.v);
    return(r);
}

inline f64x4 operator*(f64x4 a, f64x4 b)
{
    f64x4 r;
    r.v = _mm256_mul_pd(a.v, b.v);
    return(r);
}

inline f64 hadd(f64x4 a)
{
    f64 result = 0;
    f64 *Scalar = (f64 *)&a.v;
    for(u32 i = 0; i < 4; ++i)
    {
        result += Scalar[i];
    }
    return(result);
}

inline f64x4 loadu(f64 *a)
{
    f64x4 r;
    r.v = _mm256_loadu_pd(a);
    return(r);
}

f32x4 loadu(const f32 *a)
{
    f32x4 r;
    r.v = _mm_loadu_ps(a);
    return(r);
}
f64x4 F32x4ToF64x4(f32x4 a)
{
    f64x4 r;
    r.v = _mm256_cvtps_pd(a.v);
    return(r);
}


void storeu(f64 *a, f64x4 b)
{
    _mm256_storeu_pd(a, b.v);
}

f64x4 BroadcastF64(const f64 *a)
{
    f64x4 Result;
    Result.v = _mm256_broadcast_sd(a);
    return(Result);
}

inline f64x4 swap2(f64x4 a)
{
    f64x4 r;
    r.v = _mm256_permute4x64_pd(a.v, 0b01001110);
    return(r);
}
inline f64x4 swap1(f64x4 a)
{ 
    f64x4 r;
    r.v = _mm256_permute_pd(a.v, 0b0101);
    return(r);
}

void correlate(int ny, int nx, const float *data, float *result) 
{
    constexpr s32 VecDim = 4;
    const s32 VecCount = (ny + VecDim - 1) / VecDim;

    f64 *NormData0 = (f64 *)aligned_alloc(sizeof(f64),ny * nx * sizeof(f64));

   u64 StartTime = __rdtsc();

    #pragma omp parallel for schedule(dynamic)
    for(s32 Row = 0; Row < ny; ++Row)
    {
        f64 Sum = 0;
        for (s32 Col = 0; Col < nx; ++Col)
        {
            f64 val = data[nx * Row + Col];
            NormData0[nx * Row + Col] = val;
            Sum += val;
        }

        f64 Mean = Sum/nx;
        f64 SumSq = 0;
        for(s32 Col = 0; Col < nx; ++Col)
        {
            f64 val = NormData0[nx*Row + Col] - Mean;
            NormData0[nx*Row + Col] = val;
            SumSq += val*val;
        }

        f64 InvStdY = 1/sqrt(SumSq);
        for(s32 Col = 0; Col < nx; ++Col)
        {
            NormData0[nx*Row + Col] *= InvStdY;
        }
    }


    f64x4 *NormData  = (f64x4 *)aligned_alloc(sizeof(f64), VecCount * nx * sizeof(f64x4));

#pragma omp parallel for schedule(dynamic)
    for (s32 Row = 0; Row < VecCount; ++Row)
    {
        for (s32 Col = 0; Col < nx; ++Col)
        {
            f64* VecStart = (f64*)&NormData[nx * Row + Col];
            for (s32 Item = 0; Item < VecDim; ++Item)
            {
                s32 j = (Row * VecDim + Item);
                VecStart[Item] = (j < ny) ? NormData0[nx * j + Col] : 0;
            }
        }
    }

    u64 EndProc = __rdtsc();


#pragma omp parallel for schedule(dynamic)
    for (s32 Row = 0; Row < VecCount; ++Row)
    {
        for (s32 Col = Row; Col < VecCount; ++Col)
        {
            f64x4 vv000 = {};
            f64x4 vv001 = {};
            f64x4 vv010 = {};
            f64x4 vv011 = {};
            s32 k = 0;
            for (; k < nx-3; k+=3)
            {
                for(s32 i = 0; i < 3; ++i)
                {
                    constexpr int PF = 20;
                    __builtin_prefetch(&NormData[nx*Row + k + PF]);
                    __builtin_prefetch(&NormData[nx*Col + k + PF]);

                    f64x4 a000 = loadu((f64*)(NormData + nx * Row + k + i));
                    f64x4 b000 = loadu((f64*)(NormData + nx * Col + k + i));
                    f64x4 a010 = swap2(a000);
                    f64x4 b001 = swap1(b000);

                    vv000 = vv000 + a000 * b000;
                    vv001 = vv001 + a000 * b001;
                    vv010 = vv010 + a010 * b000;
                    vv011 = vv011 + a010 * b001;

                }
            }
            for (; k < nx; ++k)
            {
                    f64x4 a000 = loadu((f64*)(NormData + nx * Row + k ));
                    f64x4 b000 = loadu((f64*)(NormData + nx * Col + k ));
                    f64x4 a010 = swap2(a000);
                    f64x4 b001 = swap1(b000);

                    vv000 = vv000 + a000 * b000;
                    vv001 = vv001 + a000 * b001;
                    vv010 = vv010 + a010 * b000;
                    vv011 = vv011 + a010 * b001;
            }

            vv001 = swap1(vv001);
            vv011 = swap1(vv011);
            f64x4 vv[4] = { vv000, vv001, vv010, vv011 };


            for (s32 jb = 0; jb < VecDim; ++jb)
            {
                for (s32 ib = 0; ib < VecDim; ++ib)
                {
                    s32 i = ib + Row * VecDim;
                    s32 j = jb + Col * VecDim;
                    f64* Out = (f64*)&vv[ib ^ jb];
                    if (j < ny && i < ny)
                    {
                        result[ny * i + j] = Out[jb];
                    }
                }
            }
        }


    }


    
    u64 EndCompute = __rdtsc();
    free(NormData);
    free(NormData0);

    u64 TotalTime = EndCompute - StartTime;
    u64 PreTime = EndProc - StartTime;
    u64 CompTime = EndCompute - EndProc;
    printf("Percent Time in Proc %lld\n",PreTime * 100 / TotalTime);
    printf("Percent Time in Compute %lld\n",CompTime * 100 / TotalTime);
}
