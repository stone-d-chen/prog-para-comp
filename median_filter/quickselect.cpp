#include "math.h"
#include <random>

void swap(f32 *a, f32 *b)
{
    f32 temp = *a; //is this okay?
    *a = *b;
    *b = temp;
}

f32 *Partition(f32 *arr, u32 count, u32 index)
{
    if(count == 1 & index == 1) return(arr);
    s32 pivotIdx = rand() % count;
    f32 pivot = arr[pivotIdx];

    u32 frontIdx = 0;
    u32 backIdx = count - 1;

    //  3, 4, 1, 5, 2
    //  2, 4, 1, 5, 3
    //  2, 3, 1, 5, 4

    // 3 , 5, 4, 2, 1
    // 1, 5, 4 , 2, 3
    // 1, 2, 4, 5, 3
    
    while(frontIdx < backIdx)
    {
        while(arr[frontIdx] < pivot) {
            ++frontIdx;  //if we exit, then frontIdx points at > pivot
        }
        while(arr[backIdx] >= pivot) {
            --backIdx; //exit, points at < pivot
        }
        if(frontIdx < backIdx)
        {
            swap(arr + frontIdx++, arr + backIdx--);
        }
    }

    u32 LeftCount = frontIdx;
    u32 RightCount = count - backIdx;
    if(LeftCount >= index)
    {
        u32 newIndex = index;
        f32 *newArray = arr;
        return(Partition(newArray, LeftCount, newIndex));
    }
    else
    {
        u32 newIndex = index - LeftCount;
        f32 *newArray = arr + LeftCount;
        return(Partition(newArray, RightCount, newIndex));
    }


}


#if 1

int main()
{
    f32 test[] = {0.5, 6, 4.1, 1, 0, 0, 3.2};
    f32 *ptr = Partition(test, 7, 3 + 1);

}

#endif