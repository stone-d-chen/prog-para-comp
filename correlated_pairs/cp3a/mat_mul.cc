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

#include <intrin.h>

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
f64x4 operator-(f64x4 a, f64x4 b)
{
    f64x4 r;
    r.v = _mm256_sub_pd(a.v, b.v);
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

f64x4 BroadcastF64(const f64 *a)
{
    f64x4 Result;
    Result.v = _mm256_broadcast_sd(a);
    return(Result);
}

f64x4 swap2(f64x4 a)
{
    f64x4 r;
    r.v = _mm256_permute4x64_pd(a.v, 0b01001110);
    return(r);
}
f64x4 swap1(f64x4 a)
{ 
    f64x4 r;
    r.v = _mm256_permute_pd(a.v, 0b0101);
    return(r);
}


void correlate(int ny, int nx, const float *data, float *result) 
{
    constexpr s32 VecDim = 4;
    s32 VecCount = (ny + VecDim - 1) / VecDim;
    s32 PaddedY = VecCount*VecDim;

    f64x4 *NormData  = (f64x4 *)malloc(VecCount * nx * sizeof(f64));

    for(s32 Row = 0; Row < VecCount; ++Row)
    {
        for(s32 Col = 0; Col < nx; ++Col)
        {
            f64 *VecStart = (f64*)&NormData[nx*Row + Col];
            for(s32 Item = 0; Item < VecDim; ++Item)
            {
                s32 j = (Row * VecDim + Item);
                VecStart[Item] =  (j < nx) ? data[nx*j + Col] : 0;
            }
        }
    }

    for(s32 Row = 0; Row < VecCount; ++Row)
    {
        for(s32 Col = Row; Col < VecCount; ++Col)
        {
            f64x4 vv000 = {};
            f64x4 vv001 = {};
            f64x4 vv010 = {};
            f64x4 vv011 = {};
            for(s32 k = 0; k < nx; ++k)
            {
                f64x4 a000 = NormData[nx*Row + k];
                f64x4 a010 = swap2(a000);
                f64x4 b000 = NormData[nx*Col + k];
                f64x4 b001 = swap1(b000);

                vv000 = vv000 + a000 * b000;
                vv001 = vv001 + a000 * b001;
                vv010 = vv010 + a010 * b000;
                vv011 = vv011 + a010 * b001;
            }
            vv001 = swap1(vv001);
            vv011 = swap1(vv011);
            f64x4 vv[4] = {vv000, vv001, vv010, vv011};


            for(s32 jb = 0; jb < VecDim; ++jb)
            {
                for(s32 ib = 0; ib < VecDim; ++ib)
                {
                    s32 i = ib + Row*VecDim;
                    s32 j = jb + Col*VecDim;
                    f64 *Out = (f64*)&vv[ib^jb];
                    if(j < ny && i < ny)
                    {
                        result[ny*i + j] = Out[jb];
                    }
                }
            }
        }
    }

    
  
}

f32 data[2][3] = 
{
  1.00000000, 2.00000000, 3.0,
  3.00000000, 4.00000000, 4.0,
};
f32 dataT[3][2] = 
{
  1.00000000, 3.00000000,
  2.00000000, 4.00000000,
  3.0,         4.0,
};

f32 result[2*2];

int main()
{
    correlate(2, 3, (f32*) data, (f32*) result);
    result;
}