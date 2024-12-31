# HashMatrix
Combining features of a Matrix and Hashtable for optimized lookup and compute with the tradeoff being insert time, without the data deduplication of using both.

## Benchmark Parameters
- CPU: 24 cores @ 400 MHz
- Cache:
  - L1 Data: 32 KiB (x12)
  - L1 Instruction: 32 KiB (x12)
  - L2 Unified: 1024 KiB (x12)
  - L3 Unified: 32768 KiB (x2)

## Benchmark Results (time in microseconds)

| Operation | Size | Matrix | HashMap | HashMatrix | OptimizedHashMatrix |
|-----------|------|--------|---------|------------|-------------------|
| Insert    | 64   | 1.23   | 0.649   | 1.64       | 0.844            |
|           | 256  | 81.7   | 21.2    | 45.1       | 22.7             |
|           | 1024 | 1796   | 594     | 1190       | 610              |
| BatchInsert| 64   | 1.28   | 0.792   | 0.063      | 0.906            |
|           | 256  | 83.8   | 15.9    | 0.466      | 27.4             |
|           | 1024 | 1793   | 446     | 7.02       | 588              |
| Access    | 64   | 0.001  | 0.005   | 0.005      | 0.005            |
|           | 256  | 0.002  | 0.016   | 0.014      | 0.014            |
|           | 1024 | 0.008  | 0.187   | 0.051      | 0.240            |
| Update    | 64   | 0.001  | 0.003   | 0.018      | 0.004            |
|           | 256  | 0.002  | 0.008   | 0.055      | 0.012            |
|           | 1024 | 0.005  | 0.034   | 0.217      | 0.046            |
| Delete    | 64   | 1.32   | 0.829   | 0.096      | 1.01             |
|           | 256  | 85.2   | 20.1    | 0.541      | 27.6             |
|           | 1024 | 1852   | 543     | 7.41       | 585              |
| Search    | 64   | 0.001  | 0.005   | 0.005      | 0.005            |
|           | 256  | 0.003  | 0.014   | 0.015      | 0.019            |
|           | 1024 | 0.012  | 0.216   | 0.052      | 0.195            |
| Add       | 64   | 1.91   | 0.802   | 0.020      | 33.8             |
|           | 256  | 108    | 20.1    | 0.133      | 977              |
|           | 1024 | 2196   | 529     | 1.93       | 47229            |
| Multiply  | 64   | 102    | 0.845   | 1.71       | 3.68             |
|           | 256  | 8944   | 1075    | 27.0       | 58.3             |
|           | 1024 | 641964 | 514135  | 436        | 931              |
