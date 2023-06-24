/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/

typedef unsigned int u32;
typedef int s32;
typedef float f32;
typedef double f64;

void correlate(int ny, int nx, const float *data, float *result)
{
    for(u32 Row = 0; Row < ny; ++Row)
    {
        for(u32 Col = 0; Col < nx; ++Col)
        {
            
        }
    }
}
