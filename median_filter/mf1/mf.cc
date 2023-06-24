/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in in[x + y*nx]
- for each pixel (x, y), store the median of the pixels (a, b) which satisfy
  max(x-hx, 0) <= a < min(x+hx+1, nx), max(y-hy, 0) <= b < min(y+hy+1, ny)
  in out[x + y*nx].
*/
#include <stdlib.h>
#include "math.h"
#include "quickselect.cpp"

void mf(int ny, int nx, int hy, int hx, const float* in, float* out)
{
  f32* arr = (f32*)malloc((2 * hy + 2) * (2 * hx + 2) * sizeof(f32));
  for (s32 CenterY = 0; CenterY < ny; ++CenterY)
  {
  for (s32 CenterX = 0; CenterX < nx; ++CenterX)
  {
      // u32 CenterX = 0, CenterY = 0;

      s32 MinX = CenterX - hx; if (MinX < 0) MinX = 0;
      s32 MinY = CenterY - hy; if (MinY < 0) MinY = 0;
      s32 MaxX = CenterX + hx; if (MaxX > nx - 1) MaxX = nx - 1;
      s32 MaxY = CenterY + hy; if (MaxY > ny - 1) MaxY = ny - 1;


      // f32 *ptr = arr;
      u32 total = 0;
      for (s32 y = MinY; y <= MaxY; ++y)
      {
      for (s32 x = MinX; x <= MaxX; ++x)
        {
          arr[total++] = in[nx * y + x];
        }
      }

      f32 median = 0;
      if (total % 2 == 1)
      {
        median = *QuickSelect(arr, total, total / 2 + 1);
      }
      else
      {
        // median = *QuickSelect(arr, total, total / 2);
        // median += *QuickSelect(arr, total, total / 2 + 1);
        f32 *ptr = QuickSelect(arr, total, total / 2);
        median = *ptr;
        median += *QuickSelect(ptr + 1, total/2, 1);
        median /= 2;
      }

      out[nx * CenterY + CenterX] = median;

    }
  }
  free(arr);
}
