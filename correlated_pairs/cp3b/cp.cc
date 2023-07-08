
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

#define LANE_WIDTH 8

#if LANE_WIDTH == 8

typedef struct
{
    __m256 v;
} lane_f32;

lane_f32 operator+(lane_f32 a, lane_f32 b)
{
    lane_f32 Result;
    Result.v = _mm256_add_ps(a.v, b.v);
    return(Result);
}
lane_f32 operator-(lane_f32 a, lane_f32 b)
{
    lane_f32 Result;
    Result.v = _mm256_add_ps(a.v, b.v);
    return(Result);
}
lane_f32 operator*(lane_f32 a, lane_f32 b)
{
    lane_f32 Result;
    Result.v = _mm256_mul_ps(a.v, b.v);
    return(Result);
}

lane_f32 loadu(const f32 *a)
{
    lane_f32 Result;
    Result.v = _mm256_loadu_ps(a);
    return(Result);
}

lane_f32 swap4(lane_f32 a)
{
    lane_f32 Result;
    Result.v = _mm256_permute2f128_ps(a.v, a.v, 0b00000001);
    return(Result);
}

lane_f32 swap2(lane_f32 a)
{
    lane_f32 Result;
    Result.v = _mm256_permute_ps(a.v, 0b01001110);
    return(Result);
}

lane_f32 swap1(lane_f32 a)
{
    lane_f32 Result;
    Result.v = _mm256_permute_ps(a.v, 0b10110001);
    return(Result);
}

#endif 



void correlate(int ny, int nx, const float *data, float *result) 
{
    constexpr s32 VecDim = LANE_WIDTH;
    const s32 VecCount = (ny + VecDim - 1) / VecDim;

    f32 *NormData0 = (f32 *)aligned_alloc(sizeof(f32),ny * nx * sizeof(f32));

    u64 StartTime = __rdtsc();

    #pragma omp parallel for schedule(dynamic)
    for(s32 Row = 0; Row < ny; ++Row)
    {
        f32 Sum = 0;
        for (s32 Col = 0; Col < nx; ++Col)
        {
            f32 val = data[nx * Row + Col];
            NormData0[nx * Row + Col] = val;
            Sum += val;
        }

        f32 Mean = Sum/nx;
        f32 SumSq = 0;
        for(s32 Col = 0; Col < nx; ++Col)
        {
            f32 val = NormData0[nx*Row + Col] - Mean;
            NormData0[nx*Row + Col] = val;
            SumSq += val*val;
        }

        f32 InvStdY = 1/sqrt(SumSq);
        for(s32 Col = 0; Col < nx; ++Col)
        {
            NormData0[nx*Row + Col] *= InvStdY;
        }
    }


    lane_f32 *NormData  = (lane_f32 *)aligned_alloc(sizeof(f32), VecCount * nx * sizeof(lane_f32));

#pragma omp parallel for schedule(dynamic)
    for (s32 Row = 0; Row < VecCount; ++Row)
    {
        for (s32 Col = 0; Col < nx; ++Col)
        {
            f32* VecStart = (f32*)&NormData[nx * Row + Col];
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
            lane_f32 vv000 = {};
            lane_f32 vv001 = {};
            lane_f32 vv010 = {};
            lane_f32 vv011 = {};

            lane_f32 vv100 = {};
            lane_f32 vv101 = {};
            lane_f32 vv110 = {};
            lane_f32 vv111 = {};
            s32 k = 0;
            for (; k < nx-3; k+=3) // this is actually just ILP
            {
                for(s32 i = 0; i < 3; ++i)
                {
                    constexpr int PF = 20;
                    __builtin_prefetch(&NormData[nx*Row + k + PF]);
                    __builtin_prefetch(&NormData[nx*Col + k + PF]);

                    lane_f32 a000 = loadu((f32*)(NormData + nx * Row + k + i));
                    lane_f32 b000 = loadu((f32*)(NormData + nx * Col + k + i));
                    
                    lane_f32 a100 = swap4(a000);
                    lane_f32 a010 = swap2(a000);
                    lane_f32 a110 = swap2(a100);
                    lane_f32 b001 = swap1(b000);

                    vv000 = vv000 + a000 * b000;
                    vv001 = vv001 + a000 * b001;
                    vv010 = vv010 + a010 * b000;
                    vv011 = vv011 + a010 * b001;

                    vv100 = vv100 + a100 * b000;
                    vv101 = vv101 + a100 * b001;
                    vv110 = vv110 + a110 * b000;
                    vv111 = vv111 + a110 * b001;

                }
            }
            for (; k < nx; ++k)
            {
                    lane_f32 a000 = loadu((f32*)(NormData + nx * Row + k));
                    lane_f32 b000 = loadu((f32*)(NormData + nx * Col + k));
                    
                    lane_f32 a100 = swap4(a000);
                    lane_f32 a010 = swap2(a000);
                    lane_f32 a110 = swap2(a100);
                    lane_f32 b001 = swap1(b000);

                    vv000 = vv000 + a000 * b000;
                    vv001 = vv001 + a000 * b001;
                    vv010 = vv010 + a010 * b000;
                    vv011 = vv011 + a010 * b001;

                    vv100 = vv100 + a100 * b000;
                    vv101 = vv101 + a100 * b001;
                    vv110 = vv110 + a110 * b000;
                    vv111 = vv111 + a110 * b001;
            }

            lane_f32 vv[LANE_WIDTH] = { vv000, vv001, vv010, vv011, vv100, vv101, vv110, vv111 };
            vv001 = swap1(vv001);
            vv011 = swap1(vv011);
            vv101 = swap1(vv101);
            vv111 = swap1(vv111);


            for (s32 jb = 0; jb < VecDim; ++jb) {
                for (s32 ib = 0; ib < VecDim; ++ib) {
                    s32 i = ib + Row * VecDim;
                    s32 j = jb + Col * VecDim;
                    f32* Out = (f32*)&vv[ib ^ jb];
                    if (j < ny && i < ny) {
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
