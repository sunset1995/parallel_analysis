# Parallel Video Processing
Parallel Programming (Fall 2015) Final Project

Quick links:
[Presentation](https://github.com/sunset1995/parallel_analysis/blob/master/_presentation/Group9_Parallel-Video-Processing.pdf)
[Final Report](https://github.com/sunset1995/parallel_analysis/blob/master/_report/Group9_Parallel-Video-Processing.pdf)

## Results
### 01/03/2016
Tested environment: 4 Threads, i7-3770, 16GB, GTX670. VS12, Windows 10

**Sequential**
```
    Total: 31.893s (include time count)
    Input: 6.054s
   Output: 23.143s
Calculate: 2.613s
```

**Pthread**
```
    Total: 26.472s (include time count)
    Input: 2.528s
   Output: 22.999s
Calculate: 0.945s
```

**OpenMP**
```
    Total: 26.276s (include time count)
    Input: 2.425s
   Output: 22.887s
Calculate: 0.963s
```

**Task Parallel**
```
    Total: 24.565s (include time count)
    Input: 10.306s
   Output: 24.563s
Calculate: 3.746s
```

**CUDA**
```
    Total: 34.246s (include time count)
    Input: 4.777s
   Output: 23.699s
Calculate: 5.770s
```

**CUDA TDM**
```
    Total: 43.443s (include time count)
    Input: 4.762s
   Output: 23.437s
Calculate: 15.244s
```

### 01/03/2016
Tested environment: 2 to 4 Threads, i5-4200H, 8GB, Ubuntu 14.04

Testdata: 720*1280, 1421 frames

**Sequential**
```
    Total: 20.113s (include time count)
    Input: 5.915s
   Output: 11.867s
Calculate: 2.330s
```

**Pthread 4 threads**
```
    Total: 15.448s (include time count)
    Input: 3.009s
   Output: 11.234s
Calculate: 1.115s
```

**OpenMP 2 threads**
```
    Total: 16.137s (include time count)
    Input: 3.496s
   Output: 11.464s
Calculate: 1.177s
```

**Task Parallel 1 thread Output + 1 thread Input&Calculate**
```
    Total: 11.648s (include time count)
    Input: 6.475s
   Output: 11.648s
Calculate: 2.429s
```
