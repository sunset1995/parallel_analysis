# parallel_analysis
Parallel final research project

## Results
### 12/29/2015
Tested environment: Win 10 x64. i7-3770, 16GB, GTX670. VS12

**Sequential**
```
    Total: 5.789s (include time count)
    Input: 0.474s
   Output: 3.624s
Calculate: 0.576s
```
**pthread (C++11 thread) with 4 threads**
```
    Total: 6.896s (include time count)
    Input: 2.043s
   Output: 3.537s
Calculate: 0.165s
```
**OpenMP with 4 threads**
```
    Total: 5.055s (include time count)
    Input: 0.955s
   Output: 3.529s
Calculate: 0.571s
```
**CUDA (GTX670)**
```
    Total: 8.923s (include time count)
    Input: 0.910s
   Output: 3.530s
Calculate: 4.483s
```
