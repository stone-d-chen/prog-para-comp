/*
    Multicore via OMP

*/




#include <stdlib.h>
#include <cmath>

typedef unsigned int u32;
typedef int s32;
typedef float f32;
typedef double f64;
void correlate(int ny, int nx, const float *data, float *result) 
{
    s32 VecDim = 8;
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

    


}
