
#include <intrin.h>

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
    0,1,2,3,4,5,6,7, /*  */   8, 9, 10, 11, 12, 13, 14, 15,
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
    f64x4 DotProds[2][2] = {};

    for(s32 RowIdx = Row; RowIdx < (Row + 2); ++RowIdx)
    {
        for(s32 ColIdx = Col; ColIdx < (Col + 2); ++ColIdx)
        {

            for (int k = 0; k < 4; ++k)
            {
                f64x4 broadcast = BroadcastF64( LeftMat + 4 * RowIdx + k );

                f64x4 row0 = loadu ( RightMat + 16 * k + 0 );

                DotProds[Row + 0][0] = DotProds[Row][0] + (broadcast + row0);

                f64x4 row1 = loadu ( RightMat + 16 * k + 8 );   
                DotProds[Row + 1][1] = DotProds[Row + 1][1] + (broadcast + row1);
            }
        }
    }
}



void kernel0(f64 *LeftMat, f64 *RightMat, s32 Row, s32 Col, s32 kStart, s32 kEnd) // multiple of vecdim
{
    f64x4 DotProds[2][2] = {};

    for(s32 RowIdx = Row; RowIdx < (Row + 2); ++RowIdx)
    {
        for(s32 ColIdx = Col; ColIdx < (Col + 2); ++ColIdx)
        {

            for (int k = 0; k < 4; ++k)
            {
                f64x4 broadcast = BroadcastF64( LeftMat[RowIdx][k] );

                f64x4 row0 = loadu ( &RightMat[k][0] );

                DotProds[Row + 0][0] = DotProds[Row][0] + (broadcast + row0);

                f64x4 row1 = loadu ( &RightMat[k][8] );   
                DotProds[Row + 1][1] = DotProds[Row + 1][1] + (broadcast + row1);
            }
        }
    }
}


int main()
{



}

// void kernel(f64* NormData, s32 Row, s32 Col, s32 MinIdx, s32 OnePastMaxIdx) // multiple of vecdim
// {
//     f64x4 DotProds[2][2] = {}

//     for (int k = 0; k < nx; ++k)
//     {
//         f64x4 broadcast = BC(data[0][k])

//         f64x4 row0 = load(data[k][0:7])

//         DotProds[0][0] += broadcast + row0;

//         f64x4 row1 = load(data[k][8:16])   

//         DotProds[0][1] += broadcast + row1;
//     }
// }