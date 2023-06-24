// Instruction Level Parallelizaton
/*
  Main idea would be to have the inner loop do multiple steps at once, and save it out
  By steps I mean [nx*Row + k]*[nx*Row + k] and [nx*Row + k + 1]*[nx*Row + k + 1] can execute independently 
  We'll pad and fill with 0s to handle overlap 
*/

#include <stdlib.h>
#include <cmath>

typedef unsigned int u32;
typedef int s32;
typedef float f32;
typedef double f64;


void correlate(int ny, int nx, const float *data, float *result)
{
    const s32 VecDim = 20;
    s32 VecCount = (nx + VecDim - 1) / VecDim;
    s32 PaddedX = VecDim * VecCount;

    // malloc new dimension PaddedX * ny
    f64 *NormData = (f64 *)malloc(ny*PaddedX * sizeof(f64));
    
    for(s32 Row = 0; Row < ny; ++Row)
    {
        //loop in vectors here?
        f64 Sum = 0;
        for(s32 Col = 0; Col < PaddedX; ++Col)
        {
                    // fill 0 if past data
            f64 val = (Col < nx) ? data[nx*Row + Col] : 0;
            Sum += val;
            NormData[PaddedX*Row + Col] = val;
        }
        f64 Mean = Sum / nx;

        f64 SumSqY = 0;
        for(s32 Col = 0; Col < nx; ++Col)
        {
            f64 val = NormData[PaddedX*Row + Col] - Mean;
            NormData[PaddedX*Row + Col] = val;
            SumSqY += val*val;
        }
        f64 InvStdY = 1/sqrt(SumSqY);
        for(s32 Col = 0; Col < nx; ++Col)
        {
            NormData[PaddedX*Row + Col] *= InvStdY;
        }
    }

    for(s32 Row = 0; Row < ny; ++Row)
    {
        for(s32 Col = Row; Col < ny; ++Col)
        {
            // new inner start
            f64 DotProd[VecDim] = {};
            for(s32 VecIdx = 0; VecIdx < VecCount; ++VecIdx)
            {
                for(s32 ItemIdx = 0; ItemIdx < VecDim; ++ItemIdx)
                {
                    DotProd[ItemIdx] += NormData[PaddedX*Row + VecIdx*VecDim + ItemIdx] * NormData[PaddedX*Col + VecIdx*VecDim + ItemIdx];
                }
            }

            f64 FinalSum = 0;
            for(s32 ItemIdx = 0; ItemIdx < VecDim; ++ItemIdx)
            {
                FinalSum += DotProd[ItemIdx];
            }

            result[ny*Row + Col] = FinalSum;
        }
    }

    free(NormData);
}
