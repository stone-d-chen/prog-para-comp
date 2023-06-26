#ifndef MATH_H
#define MATH_H


typedef unsigned int u32;
typedef int s32;
typedef float f32;
typedef double f64;


#endif

/*
    a function that will create a partial set of dot prods based on the VecIdx
    X[x:x+3][MinVec*VecDim + MaxVec*VecDim]
    Out[x,y] = X[x][k] * X'[k][y] for all k
    Out[x,y + 1] = X[x][k] * X'[k][y+1]

*/

void kernel(f64 *NormData, s32 Row, s32 Col,
            s32 MinIdx, s32 OnePastMaxIdx) // multiple of vecdim
            {
                f64x4 DotProds[3][3] = {};
                const s32 VecDim = 8;

                for(s32 Idx = MinIdx; Idx < OnePastMaxIdx; ++Idx)
                {
                    f64x4 x = BroadcastF64( NormData[PaddedX*Row + Idx] );

                    for(s32 j = 0; j < 3; ++j)
                    {


                        f64x4 y = loadu(NormData[PaddedX*(Col + Idx + j*VecDim)] )

                    }
                }
            }

//C[x:x+3][y:y+3]

//


// update 6x16 submatrix C[x:x+6][y:y+16]
// using A[x:x+6][l:r] and B[l:r][y:y+16]
void kernel(float *a, vec *b, vec *c, int x, int y, int l, int r, int n)
{
    vec t[6][2]{}; // will be zero-filled and stored in ymm registers

    for (int k = l; k < r; k++)
    {
        for (int i = 0; i < 6; i++)
        {
            // broadcast a[x + i][k] into a register
            vec alpha = vec{} + a[(x + i) * n + k]; // converts to a broadcast
            
            // multiply b[k][y:y+16] by it and update t[i][0] and t[i][1]
            for (int j = 0; j < 2; j++)
                t[i][j] += alpha * b[(k * n + y) / 8 + j]; // converts to an fma
        }
    }

    // write the results back to C
    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 2; j++)
            c[((x + i) * n + y) / 8 + j] += t[i][j];
}

