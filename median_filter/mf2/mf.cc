/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in in[x + y*nx]
- for each pixel (x, y), store the median of the pixels (a, b) which satisfy
  max(x-hx, 0) <= a < min(x+hx+1, nx), max(y-hy, 0) <= b < min(y+hy+1, ny)
  in out[x + y*nx].
*/

/*
fix quick select
might need to just call it twice


*/
#include <algorithm>
#include <math.h>
#include <stdint.h>
// #include "quickselect.cpp"
typedef float f32;
typedef uint32_t u32;

void swap(f32 &a, f32 &b)
{
  f32 temp = a;
  a = b;
  b = temp;
}



void mf(int ny, int nx, int hy, int hx, const float* in, float* out)
{

  #pragma omp parallel for
  for (int CenterY = 0; CenterY < ny; ++CenterY)
  {
    f32* arr = (f32*)malloc(4 * (hy + 1) *  (hx + 1) * sizeof(f32));
    for (int CenterX = 0; CenterX < nx; ++CenterX)
    {
      int MinX = CenterX - hx; if (MinX < 0) MinX = 0;
      int MinY = CenterY - hy; if (MinY < 0) MinY = 0;
      int OnePastMaxX = CenterX + hx + 1; if (OnePastMaxX > nx) OnePastMaxX = nx;
      int OnePastMaxY = CenterY + hy + 1; if (OnePastMaxY > ny) OnePastMaxY = ny;

      f32* pt = arr;
      int total = 0;
      for (int y = MinY; y < OnePastMaxY; ++y)
      {
        for (int x = MinX; x < OnePastMaxX; ++x)
        {
          *pt++ = in[nx * y + x];
          ++total;
        }
      }

      // std::sort(arr, arr + total);

      f32 median = 0;
      if (total % 2 == 1)
      {
        // median = *Partition(arr, total, (total)/2 + 1);
        std::nth_element(arr, arr + total/2, arr + total);
        median = arr[total/2];
      }
      else
      {
        // f32 *ptr = Partition(arr, total, (total)/2  );
        // median = *ptr;
        // median += *Partition(arr, total, (total)/2 + 1 );
        // median /= 2;

        std::nth_element(arr, arr + (total/2 - 1), arr + total);
        median = arr[total/2 - 1] ;
        std::nth_element(arr + total/2 - 1, arr, arr + total);
        median += arr[total/2 ] ;
        median /= 2;

      }
      out[nx * CenterY + CenterX] = median;

    }
    free(arr);  
  }
}


#if 0

f32 test[5][5] = { 
  {0.0, 0.0, 0.0, 0.0, 0.0},
  {0.0, 1.0, 1.0, 1.0, 0.0},
  {0.0, 1.0, 1.0, 1.0, 0.0},
  {0.0, 1.0, 1.0, 1.0, 0.0},
  {0.0, 0.0, 0.0, 0.0, 0.0} };

f32 out[25] = {};
int main()
{
  // f32 test[] = {3,5,4,5.6,0.5,0.4};
  f32 test[] = {0,0,0,1,0,1};
  //pivot 1, does not work
  // f32 test[] = {0,1,1,0,1,1,0,1,1};

  f32 *val = Partition(test, 6, 4);
  // f32 *val2 = QuickSelect(test, 6, 6/2+1);

  mf(5,5,1,1, (f32*)test,out);

}

#endif  