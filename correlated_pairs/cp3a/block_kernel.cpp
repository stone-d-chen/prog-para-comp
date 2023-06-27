
#include <intrin.h>
#include <math.h>

typedef unsigned int u32;
typedef int s32;
typedef float f32;
typedef double f64;


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

f64x4 BroadcastF64(f64 *a)
{
    f64x4 Result;
    Result.v = _mm256_broadcast_sd(a);
    return(Result);
}

f64 LeftMat[16][4] = 
{
    0,1,2,3,
    4,5,6,7,
    8,9,10,11,
    12,13,14,15,
    0,1,2,3,
    4,5,6,7,
    8,9,10,11,
    12,13,14,15,
    0,1,2,3,
    4,5,6,7,
    8,9,10,11,
    12,13,14,15,
    0,1,2,3,
    4,5,6,7,
    8,9,10,11,
    12,13,14,15,
};

f64 RightMat[4][16] = 
{
    0,1,2,3,4,5,6,7, /*  */   8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, /**/ 24, 25, 26, 27, 28, 29, 30, 31,
    0,1,2,3,4,5,6,7, /*  */   8, 9, 10, 11, 12, 13, 14, 15,
    0,1,2,3,4,5,6,7, /*  */   8, 9, 10, 11, 12, 13, 14, 15,
};


f64 LeftMat2[4][8] =
{
    0,1,2,3,4,5,6,7,
    8,9,10,11,12,13,14,15,
    16,17,18,19,20,21,22,23,
//  0,4,0,4,0,4,0,4
    24,25,26,27,28,29,30,31
};
f64 RightMat2[8][4] =
{
    0,1,2,3,
    4,5,6,7,
    0,1,2,3,
    4,5,6,7,
    0,1,2,3,
    4,5,6,7,
    0,1,2,3,
    4,5,6,7,
    // 8,9,10,11,12,13,14,15,
    // 16,17,18,19,20,21,22,23,
    // 24,25,26,27,28,29,30,31
};


f64 LeftMat3[4][4] =
{
    0,1,2,4,
    0,1,2,4,
    0,1,2,4,
    0,1,2,4,
};
f64 RightMat3[4][4] =
{
    0,1,2,4,
    0,1,2,4,
    0,1,2,4,
    0,1,2,4,
};

f64 result[16][16] = {};
f64 result2[4][4] = {};



void kernel(f64 *LeftMat, f64 *RightMat, f64 *Result,
            s32 Row, s32 Col,
            s32 kStart, s32 kEnd,
            s32 DimInner, s32 DimOuter) // multiple of vecdim
{
    const s32 VecWidth = 4;
    const s32 OutDim = 3;
    const s32 ny = 16;
    const s32 nx = 4;

    f64x4 DotProds[OutDim][OutDim] = {};

    for (int k = kStart; k < kEnd; ++k)
    {
            for(s32 i = 0; i < OutDim; ++i)
            {
                f64x4 broadcast = BroadcastF64( LeftMat + DimInner * (Row + i) + k );

                for(s32 j = 0; j < OutDim; ++j)
                {
                    f64x4 row = loadu ( RightMat + (DimOuter * k) + (Col + j * VecWidth) );
                    DotProds[i][j] = DotProds[i][j] + (broadcast * row);
                }
            }
    }

    f64 *dp = (f64 *)DotProds;
    
    for(s32 i  = 0; i < OutDim; ++i)
    {
        for(s32 j = 0; j < VecWidth * OutDim; ++j)
        {
            if((Row + i < DimOuter) && (Col + j < DimOuter))
            {
                Result[DimOuter * (Row + i) + Col + j] = dp[VecWidth*OutDim * i + j];
            }
        }
    }
    
}

int main()
{
    for(int x = 0; x < 4; x += 3)
        for(int y = 0; y < 16; y+=12)
            kernel((f64*)LeftMat2, (f64*) RightMat2, (f64 *)result2, x, y, 0, 8, 4, 8  );

    for(int x = 0; x < 4; x += 3)
        for(int y = 0; y < 16; y+=12)
            kernel((f64*)LeftMat3, (f64*) RightMat3, (f64 *)result2, x, y, 0, 4, 4, 4  );

    kernel((f64*)LeftMat, (f64*) RightMat, (f64 *)result, 0, 0, 0, 4, 16, 4);

}
