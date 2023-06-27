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

2) blocking ILP 3x3
    benchmarks/1.txt                  0.008s  pass
    benchmarks/2a.txt                 0.164s  pass
    benchmarks/2b.txt                 0.162s  pass
    benchmarks/2c.txt                 0.143s  pass
    benchmarks/2d.txt                 0.158s  pass
    benchmarks/3.txt                  4.542s  pass
    benchmarks/4.txt                 15.516s  pass

gflops falling off suggests that I'm running into memory issues
    benchmarks/1	0.017011 s	1,004,000,000	59.0
    the input contains 1000 × 1000 pixels, and the output should contain 1000 × 1000 pixels
    benchmarks/2a	0.161448 s	16,016,000,000	99.2
    the input contains 4000 × 1000 pixels, and the output should contain 4000 × 4000 pixels
    benchmarks/2b	0.163605 s	16,016,000,000	97.9
    the input contains 4000 × 1000 pixels, and the output should contain 4000 × 4000 pixels
    benchmarks/2c	0.164088 s	15,991,989,003	97.5
    the input contains 3999 × 999 pixels, and the output should contain 3999 × 3999 pixels
    benchmarks/2d	0.157263 s	16,040,029,005	102.0
    the input contains 4001 × 1001 pixels, and the output should contain 4001 × 4001 pixels
    benchmarks/3	3.441109 s	216,144,000,000	62.8
    the input contains 6000 × 6000 pixels, and the output should contain 6000 × 6000 pixels
    benchmarks/4	12.329532 s	729,324,000,000	59.2
    the input contains 9000 × 9000 pixels, and the output should contain 9000 × 9000 pixels

3) schedule dynamic ... wow ...
    benchmarks/1.txt                  0.014s  pass
    benchmarks/2a.txt                 0.083s  pass
    benchmarks/2b.txt                 0.079s  pass
    benchmarks/2c.txt                 0.085s  pass
    benchmarks/2d.txt                 0.081s  pass
    benchmarks/3.txt                  1.893s  pass
    benchmarks/4.txt                  8.297s  pass

benchmarks/4	5.547588 s	729,324,000,000	131.5
the input contains 9000 × 9000 pixels, and the output should contain 9000 × 9000 pixels


*/


#include <stdlib.h>
#include <cmath>
#include <cstring>
#include <cstdlib>

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

void correlate(int ny, int nx, const float *data, float *result) 
{
    const s32 VecDim = 4;
    s32 VecCount = (nx + VecDim - 1) / VecDim;
    s32 PaddedX = VecDim * VecCount;

    const s32 BlockDimY = 3;
    const s32 BlockDimX = 3;
    s32 BlockCountY = (ny + BlockDimY - 1) / BlockDimY;
    s32 PaddedY = BlockCountY * BlockDimY; 

    //aligned vs unaligned doesn't seem to do much
    f64 *NormData = (f64 *)aligned_alloc(sizeof(f64),PaddedY * PaddedX * sizeof(f64));
    // f64 *NormData = (f64 *)malloc(PaddedY * PaddedX * sizeof(f64));

    #pragma omp parallel for
    for(s32 Row = 0; Row < ny; ++Row)
    {
        // f64 Sum = 0;
        // for(s32 Col = 0; Col < PaddedX; ++Col)
        // {
        //     f64 val = (Col < nx) ? data[nx*Row + Col] : 0;
        //     NormData[PaddedX*Row + Col] = val;
        //     Sum += val;
        // }
        f64x4 Sum = {};
        s32 Col = 0;    
        for(; Col < nx - 4; Col+=4)
        {
            f32x4 f32val = loadu(data + nx*Row + Col);
            f64x4 f64val = F32x4ToF64x4(f32val);
            Sum = Sum + f64val;
            storeu(&NormData[PaddedX*Row + Col], f64val);
        }
        f64 Mean = hadd(Sum);
        for(; Col < PaddedX; ++Col)
        {
            f64 val = (Col < nx) ? data[nx*Row + Col] : 0;
            NormData[PaddedX*Row + Col] = val;
            Mean += val;
        }
        Mean/=nx;

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


    #pragma omp parallel for schedule(dynamic)
    for(s32 Row = 0; Row < ny; Row+=BlockDimY)
    {
        for(s32 Col =  Row; Col < ny; Col+=BlockDimX)
        {
            f64x4 DotProds[BlockDimY][BlockDimX] = {};

                for(s32 VecIdx = 0; VecIdx < PaddedX; VecIdx += VecDim)
                {
                    for(s32 i = 0; i < BlockDimY; ++i)
                    {
                        for(s32 j = 0; j < BlockDimX; ++j)
                        {
                                constexpr int PF = 20;
                                __builtin_prefetch(&NormData[PaddedX*(Row + i) + VecIdx + PF]);
                                // __builtin_prefetch(&NormData[PaddedX*(Col + i) + VecIdx + PF]);
                                f64x4 x = loadu((f64*)(NormData + PaddedX*(Row + i) + VecIdx));
                                f64x4 y = loadu((f64*)(NormData + PaddedX*(Col + j) + VecIdx));
                                DotProds[i][j] = DotProds[i][j] + (x * y);
                        }
                    }
                
                }
            

            for(s32 i = 0; i < BlockDimY; ++i)
            {
                for(s32 j = 0; j < BlockDimX; ++j)
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
