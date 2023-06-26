
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



f64 data[16][4] = 
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

f64 data2[4][16] = 
{
    0,1,2,3,4,5,6,7, /*  */   8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, /**/ 24, 25, 26, 27, 28, 29, 30, 31,
    0,1,2,3,4,5,6,7, /*  */   8, 9, 10, 11, 12, 13, 14, 15,
    0,1,2,3,4,5,6,7, /*  */   8, 9, 10, 11, 12, 13, 14, 15,
};

f64 result[16][16] = {};

f64x4 BroadcastF64(f64 *a)
{
    f64x4 Result;
    Result.v = _mm256_broadcast_sd(a);
    return(Result);
}


void kernel(f64 *LeftMat, f64 *RightMat, s32 Row, s32 Col, s32 kStart, s32 kEnd) // multiple of vecdim
{
    const s32 VecWidth = 4;
    f64x4 DotProds[2][2] = {};

    for(s32 i = 0; i < 2; ++i)
    {

            for (int k = 0; k < 4; ++k)
            {
                f64x4 broadcast = BroadcastF64( LeftMat + 4 * (Row + i) + k );

                for(s32 j = 0; j < 2; ++j)
                {
                    f64x4 row0 = loadu ( RightMat + (16 * k) + (Col + j * 4) );
                    DotProds[i][j] = DotProds[i][j] + (broadcast + row0);
                }
            }
    }
}

int main()
{

    kernel((f64*)data, (f64*) data2, 0, 0, 0, 4 );

}
