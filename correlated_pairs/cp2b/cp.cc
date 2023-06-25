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
    const s32 VecDim = 8;
    s32 VecCount = (nx + VecDim - 1) / VecDim;
    s32 PaddedX = VecDim * VecCount;

    f64 *NormData = (f64 *)malloc(ny * PaddedX * sizeof(f64));

    #pragma omp parallel for
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


    // the output dim
    #pragma omp parallel for
    for(s32 Row = 0; Row < ny; ++Row)
    {
        for(s32 Col =  Row; Col < ny; ++Col)
        {
            f64 DotProds[VecDim] = {};
            for(s32 VecIdx = 0; VecIdx < VecCount; ++VecIdx)
            {
                for(s32 ItemIdx = 0; ItemIdx < VecDim; ++ItemIdx)
                {
                    DotProds[ItemIdx] += NormData[PaddedX*Row + VecIdx*VecDim + ItemIdx]*NormData[PaddedX*Col + VecIdx*VecDim + ItemIdx];
                }
            }
            f64 FinalSum = 0;
            for(s32 ItemIdx = 0; ItemIdx < VecDim; ++ItemIdx)
            {
                FinalSum += DotProds[ItemIdx];
            }
            result[ny*Row + Col] = FinalSum;
        }
    }
    free(NormData);
}
