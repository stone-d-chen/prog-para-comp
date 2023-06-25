# Programming Parallel Computers

## Todos (potentially)

-   Figure out how to check the compiler type so I don't have to keep re-doing the intrinsics file switching gcc/msvc

### Correlated Pairs

-   cp3a, look into blocking

### Median Filter

-   Hand rolled, median of medians
-   convert recursive -\> iterative
-   refactor quicksort into a separate partition function

## Updates

2023/25/6

-   cp3a blocking is working but still really slow, the gflops going down suggests I'll have to do some sort of blocking to get this to work

```{=html}
<!-- -->
```
-   cp2b omp parallel

-   cp2c was confused why `VecNormData[PaddedX*Row + VecIdx]` wasn't working but the array width now changes to `VecNormData[VecCount*Row + VecIdx]` ...just kidding this actually crashes for some reason? Unaligned loads?

2023/24/6

-   Mf2 OMP parallel version, currently fastest solution, no idea why though...
-   cp1, cp2, sequential and ILP, tried ILP on the processing phase, seems to slow things down

2023/23/6

-   Median Filter sequential, had some issues getting QuickSelect working, should do a more in-depth write up
-   Turns out I could just use nth_element which is way faster

## Sources

-   2D median filter w/ Linear Time Median Find using QuickSelect
    -   Quicksort partition, <https://en.wikipedia.org/wiki/Quicksort>

    -   QuickSelect, <https://en.wikipedia.org/wiki/Quickselect>

Based on [Jukka Suomela's](https://jukkasuomela.fi/) materials at <https://ppc.cs.aalto.fi/ack/> Creative Commons [**Attribution 4.0 International**](https://creativecommons.org/licenses/by/4.0/) (CC BY 4.0).
