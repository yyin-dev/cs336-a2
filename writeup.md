## problem(benchmarking_script)

(a) `benchmark.py` and `benchmark_sweep.py`. See results in `data/e2e_benchmark`. 

(b) Backwards takes about 2x of forward time. The variability is very small. Std/mean is usually under 1%. 

(b) With zero warmup, the variability is much higher. Std/mean can get 1.8 for small runs. Potential factors: sending instructions to GPU, loading kernels into memory, caching, etc.

With one warmup step, the variability is much lower and very close to doing five warmup steps. However, the variability is still slightly higher. Probably due to caching?