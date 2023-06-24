/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/

#include <stdlib.h>
#include <cmath>

typedef unsigned int u32;
typedef int s32;
typedef float f32;
typedef double f64;

void correlate(const int ny, const int nx, const float *data, float *result)
{
    f64 *NormX = (f64 *)malloc(ny*nx*sizeof(f64));

    // normalization step
    for(u32 Row = 0; Row < ny; ++Row)
    {
        f64 Sum = 0;
        for(u32 Col = 0; Col < nx; ++Col)
        {
            f64 val = data[nx*Row + Col];
            Sum += val;
            NormX[nx*Row + Col] = val;
        }
        f64 Mean = Sum / nx; 

        f64 SumSqY = 0;
        for(u32 Col = 0; Col < nx; ++Col)
        {
            f64 val = NormX[nx*Row + Col];
            SumSqY += val * val;
            NormX[nx*Row + Col] = val - Mean;
        }
        f64 StdY = 1/sqrt(SumSqY);

        for(u32 Col = 0; Col < nx; ++Col)
        {
            NormX[nx*Row + Col] /= StdY;
        }
    }
}
