## problem(benchmarking_script)

(a) `benchmark.py` and `benchmark_sweep.py`. See results in `data/e2e_benchmark`. Benchmarked on A100 with 40G RAM.

(b) Backwards takes about 2x of forward time. The variability is very small. Std/mean is usually under 1%. 

(b) With zero warmup, the variability is much higher. Std/mean can get 1.8 for small runs. Potential factors: sending instructions to GPU, loading kernels into memory, caching, etc.

With one warmup step, the variability is much lower and very close to doing five warmup steps. However, the variability is still slightly higher. Probably due to caching?

## problem (nsys_profiles)

See `data/nsys_profiles`. 

To apply filter in the GUI, 

* To filter a single NVTX range: right click on the range and select "Apply filter".
* To filter more flexibly, select a range, right click and select "Filter and Zoom in".

Analysis below is based on `data/nsys_profiles/d_model_1600_layers_48_heads_25_batch_4_ctx_256.nsys-rep`. GPT-2 XL with context length of 256.

(a) Roughly matches.

(b) During forward, `ampere_sgemm_128x64_tn` and `ampere_sgemm_32x128_tn` each takes around 40%. The two are still the two kernels taking most runtime if we do both forward and backward, but less significantly.

Forward only:

| Time  | Total Time | Instances | Avg        | Med        | Min        | Max       | StdDev     | Name                                                         |
| ----- | ---------- | --------- | ---------- | ---------- | ---------- | --------- | ---------- | ------------------------------------------------------------ |
| 43.0% | 74.778 ms  | 153       | 488.742 μs | 310.056 μs | 308.200 μs | 1.223 ms  | 363.704 μs | ampere_sgemm_128x64_tn                                       |
| 41.8% | 72.688 ms  | 60        | 1.211 ms   | 1.211 ms   | 1.210 ms   | 1.214 ms  | 873 ns     | ampere_sgemm_32x128_tn                                       |
| 2.5%  | 4.264 ms   | 368       | 11.585 μs  | 12.768 μs  | 8.480 μs   | 14.273 μs | 2.116 μs   | void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3) |
| 2.0%  | 3.407 ms   | 60        | 56.786 μs  | 56.753 μs  | 51.650 μs  | 62.081 μs | 4.723 μs   | void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, std::array<char *, (unsigned long)3>>(int, T2, T3) |

Backward only:

| Time  | Total Time | Instances | Avg        | Med        | Min        | Max        | StdDev     | Name                                                         |
| ----- | ---------- | --------- | ---------- | ---------- | ---------- | ---------- | ---------- | ------------------------------------------------------------ |
| 16.2% | 95.414 ms  | 79        | 1.208 ms   | 1.199 ms   | 1.199 ms   | 1.865 ms   | 74.937 μs  | ampere_sgemm_128x32_sliced1x4_nn                             |
| 15.7% | 92.405 ms  | 79        | 1.170 ms   | 1.162 ms   | 1.161 ms   | 1.787 ms   | 70.332 μs  | void cutlass::Kernel2<cutlass_80_simt_sgemm_128x32_8x5_nt_align1>(T1::Params) |
| 11.8% | 69.320 ms  | 153       | 453.072 μs | 453.068 μs | 452.523 μs | 453.836 μs | 279 ns     | void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_align1>(T1::Params) |
| 7.9%  | 46.782 ms  | 153       | 305.764 μs | 305.960 μs | 304.104 μs | 306.920 μs | 688 ns     | ampere_sgemm_64x32_sliced1x4_nt                              |
| 7.7%  | 45.582 ms  | 39        | 1.169 ms   | 1.170 ms   | 1.163 ms   | 1.172 ms   | 2.413 μs   | ampere_sgemm_128x32_nn                                       |
| 7.7%  | 45.167 ms  | 88        | 513.256 μs | 310.168 μs | 308.264 μs | 1.793 ms   | 394.764 μs | ampere_sgemm_128x64_tn                                       |
| 7.7%  | 45.151 ms  | 39        | 1.158 ms   | 1.157 ms   | 1.156 ms   | 1.162 ms   | 1.371 μs   | ampere_sgemm_32x128_nt                                       |
| 7.4%  | 43.607 ms  | 36        | 1.211 ms   | 1.211 ms   | 1.210 ms   | 1.213 ms   | 720 ns     | ampere_sgemm_32x128_tn                                       |
| 4.1%  | 24.194 ms  | 604       | 40.055 μs  | 13.360 μs  | 8.256 μs   | 213.509 μs | 45.027 μs  | void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 7)]::operator ()() const::[lambda(float) (instance 1)]>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3) |
| 3.2%  | 18.580 ms  | 582       | 31.924 μs  | 13.568 μs  | 1.824 μs   | 63.266 μs  | 25.622 μs  | void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, std::array<char *, (unsigned long)3>>(int, T2, T3) |

Forward + Backward:

| Time  | Total Time | Instances | Avg        | Med        | Min        | Max        | StdDev     | Name                                                         |
| ----- | ---------- | --------- | ---------- | ---------- | ---------- | ---------- | ---------- | ------------------------------------------------------------ |
| 15.7% | 119.947 ms | 241       | 497.705 μs | 310.152 μs | 308.136 μs | 1.793 ms   | 374.705 μs | ampere_sgemm_128x64_tn                                       |
| 15.2% | 116.300 ms | 96        | 1.211 ms   | 1.211 ms   | 1.210 ms   | 1.214 ms   | 826 ns     | ampere_sgemm_32x128_tn                                       |
| 12.5% | 95.414 ms  | 79        | 1.208 ms   | 1.199 ms   | 1.199 ms   | 1.865 ms   | 74.937 μs  | ampere_sgemm_128x32_sliced1x4_nn                             |
| 12.1% | 92.405 ms  | 79        | 1.170 ms   | 1.162 ms   | 1.161 ms   | 1.787 ms   | 70.332 μs  | void cutlass::Kernel2<cutlass_80_simt_sgemm_128x32_8x5_nt_align1>(T1::Params) |
| 9.1%  | 69.320 ms  | 153       | 453.072 μs | 453.068 μs | 452.523 μs | 453.836 μs | 279 ns     | void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_align1>(T1::Params) |
| 6.1%  | 46.782 ms  | 153       | 305.764 μs | 305.960 μs | 304.104 μs | 306.920 μs | 688 ns     | ampere_sgemm_64x32_sliced1x4_nt                              |
| 6.0%  | 45.582 ms  | 39        | 1.169 ms   | 1.170 ms   | 1.163 ms   | 1.172 ms   | 2.413 μs   | ampere_sgemm_128x32_nn                                       |
| 5.9%  | 45.151 ms  | 39        | 1.158 ms   | 1.157 ms   | 1.156 ms   | 1.162 ms   | 1.371 μs   | ampere_sgemm_32x128_nt                                       |
| 3.4%  | 26.150 ms  | 788       | 33.184 μs  | 12.512 μs  | 8.256 μs   | 213.509 μs | 41.345 μs  | void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 7)]::operator ()() const::[lambda(float) (instance 1)]>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3) |
| 2.9%  | 21.983 ms  | 642       | 34.241 μs  | 51.873 μs  | 1.824 μs   | 63.266 μs  | 25.480 μs  | void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, std::array<char *, (unsigned long)3>>(int, T2, T3) |

(c) Non-matmul kernels taking non-trivial amount of time during forward. Mostly element-wise kernels.

| Time | Total Time | Instances | Avg       | Med       | Min       | Max       | StdDev   | Name                                                         |
| ---- | ---------- | --------- | --------- | --------- | --------- | --------- | -------- | ------------------------------------------------------------ |
| 2.5% | 4.291 ms   | 370       | 11.598 μs | 12.768 μs | 8.448 μs  | 14.368 μs | 2.121 μs | void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3) |
| 2.0% | 3.403 ms   | 60        | 56.713 μs | 56.578 μs | 51.777 μs | 61.890 μs | 4.694 μs | void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, std::array<char *, (unsigned long)3>>(int, T2, T3) |
| 1.1% | 1.937 ms   | 182       | 10.644 μs | 9.792 μs  | 9.248 μs  | 15.328 μs | 1.985 μs | void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 7)]::operator ()() const::[lambda(float) (instance 1)]>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3) |

(d) During forward pass, more than 85% of total time is spent on matmul.

For one complete training step (forward + backward + optimizer step), 

| Time  | Total Time | Instances | Avg        | Med        | Min        | Max        | StdDev     | Name                                                         |
| ----- | ---------- | --------- | ---------- | ---------- | ---------- | ---------- | ---------- | ------------------------------------------------------------ |
| 11.4% | 119.939 ms | 241       | 497.670 μs | 310.056 μs | 307.816 μs | 1.793 ms   | 374.728 μs | ampere_sgemm_128x64_tn                                       |
| 11.1% | 117.007 ms | 97        | 1.206 ms   | 1.199 ms   | 1.199 ms   | 1.865 ms   | 67.612 μs  | ampere_sgemm_128x32_sliced1x4_nn                             |
| 11.0% | 116.300 ms | 96        | 1.211 ms   | 1.211 ms   | 1.210 ms   | 1.214 ms   | 932 ns     | ampere_sgemm_32x128_tn                                       |
| 10.8% | 113.321 ms | 97        | 1.168 ms   | 1.162 ms   | 1.161 ms   | 1.787 ms   | 63.457 μs  | void cutlass::Kernel2<cutlass_80_simt_sgemm_128x32_8x5_nt_align1>(T1::Params) |
| 8.3%  | 86.996 ms  | 192       | 453.101 μs | 453.067 μs | 452.524 μs | 454.220 μs | 317 ns     | void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_align1>(T1::Params) |
| 6.7%  | 70.438 ms  | 2431      | 28.975 μs  | 11.488 μs  | 1.888 μs   | 147.235 μs | 35.678 μs  | void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::array<char *, (unsigned long)3>>(int, T2, T3) |
| 5.8%  | 61.552 ms  | 2468      | 24.939 μs  | 14.400 μs  | 1.792 μs   | 96.675 μs  | 25.249 μs  | void at::native::vectorized_elementwise_kernel<(int)4, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, std::array<char *, (unsigned long)2>>(int, T2, T3) |
| 5.6%  | 58.716 ms  | 192       | 305.813 μs | 305.928 μs | 304.520 μs | 306.856 μs | 551 ns     | ampere_sgemm_64x32_sliced1x4_nt                              |
| 5.3%  | 56.105 ms  | 48        | 1.169 ms   | 1.170 ms   | 1.162 ms   | 1.172 ms   | 2.306 μs   | ampere_sgemm_128x32_nn                                       |
| 5.3%  | 55.566 ms  | 48        | 1.158 ms   | 1.157 ms   | 1.156 ms   | 1.162 ms   | 1.331 μs   | ampere_sgemm_32x128_nt                                       |

Matmul kernels sum to around 68%. So fraction of time spent on matmul is lower.

| Phase     | Dominant Ops              | Notes                             |
| --------- | ------------------------- | --------------------------------- |
| Forward   | matmuls                   | Efficient fused kernels           |
| Backward  | matmuls + elementwise ops |                                   |
| Optimizer | mostly elementwise ops    | Low in FLOPs, but high in runtime |

Forward is dominated by matmul. 

Backward pass consists of more element-wise ops. Even though gradients of matmuls are still matmuls, gradients of activations (e.g. ReLU, GeLU, softmax) are element-wise, gradients of residual connections are element-wise adds, etc.

Optimizer step usually involves very little matmuls and is dominated by element-wise ops. Take Adam for example, Given parameter θ, gradient g, moments m, v: 

```
m ← β₁ * m + (1 - β₁) * g      # 1st moment (mean)
v ← β₂ * v + (1 - β₂) * g²     # 2nd moment (variance)
θ ← θ - α * m / (sqrt(v) + ε)  # parameter update
```

All of these are element-wise ops.

GPT states matmul's time fraction should be 80%-90% for inference, and 50%-70% for full training steps.

(b) Looking at one `CausalMultiHeadSelfAttention` NVTX range:![image-20250803150116427](https://raw.githubusercontent.com/yyin-dev/image_cloud/main/Picsee/image-20250803150116427_W9GkHh.jpeg)

Time ratio:

* Entire range: 2.001ms
* Softmax range: 122.885μs, about 6.1% of the attention calculation. 

The FLOP of softmax should be negligible compared to attention caluation, but the time ratio is not!

## Problem(mixed_precision_accumulation)

See `precision.py`.

```python
import torch

# 1. Always use f32
s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float32)
print(s)  # 10.0001

# 2. Always use f16
s = torch.tensor(0, dtype=torch.float16)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16)
print(s)  # 9.9531

# 3. Accumulate results in f32, intermediate in f16
s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16)
print(s)  # 10.0021

# 4. Accumulate results in f32, intermediate in f16. (Same as 3)
s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    x = torch.tensor(0.01, dtype=torch.float16)
    s += x.type(torch.float32)
print(s)  # 10.0021

```

## Problem (benchmarking_mixed_precision)

(a) With autocasting,

* model parameters are still in float32. 
* The output of the feed-forward layer is float16
* The output of the layer norm is **float32**. 
* The model's outputs are in float16
* the loss is in float16
* The gradients are in float32

(b) For each input tensor, LayerNorm computes the tensor-level (not batch level) mean and variance, and then normalize each input through scaling the shifting. The output tensor has mean of 0 and variance of 1.

The normalization bit can be sensitive to mixed precison. The normalization is basically doing $$\frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$$, where both $$x - \mu$$ and $$\sigma$$ can be small numbers. Also, $$\epsilon$$ is a small constant for numerical stability, but float16 might not have enough precision for it.

Float16 has more precision bits but fewer exponent bits, so it has higher precision but narrower range. Bfloat16 has less precision bits but more exponent bits, so it has lower precision but wider range.

Bfloat is better at representing small values, so maybe it's less necessary to remain in float32?

(c) See data in `data/e2e_benchmark/forward_backward_and_optimizer` and `data/e2e_benchmark/forward_backward_and_optimizer_mixed_precision`

For small runs, mixed precision is even slower, probably because of overhead from mixed precision. Example: GPT-2 small w/ context length of 128, 0.068 -> 0.073.

For larger runs, mixed precision is faster. Example: GPT-2 XL w/ context length of 256, 1.09 -> 0.44.

## Problem (memory_profilng)

My A100 has only 40G of memory, so I profiled forward pass and backpass and optimizer step of the GPT-XL model. For forward pass, I profiled contxt length of 128, 256, and 512. For backward pass, I profiled 128 and 256. See `data/memory_profiles`.

(a) For context length of 256:

Forward only: 

![image-20250803170837588](https://raw.githubusercontent.com/yyin-dev/image_cloud/main/Picsee/image-20250803170837588_mFKweK.jpeg)

Full training step:

![image-20250803171305699](https://raw.githubusercontent.com/yyin-dev/image_cloud/main/Picsee/image-20250803171305699_myUtVk.jpeg)

Observations:

* Based on the shape of forward-only, we can tell in the full training image, the plateau after each spike is the backward phase.

(b) Peak memory usage:

* Forward-only:
  * 128: 13G
  * 256: 20G
  * 512: 37G
* Full training step:
  * 128: 32G
  * 256: 37G

(c) Mixed-precision memory usage:

* forward-only:
  * 128: around 15G
  * 256: around 19G
  * 512: around 31G
* full training step:
  * 128: 36G
  * 512: 37G

Sometimes, mixed-precision reduce memory usage, because tensors are cast to lower precision, there's therefore smaller activation, etc. It can save a significant amount of RAM, e.g. forward-only context length 512, 37G -> 31G.

However, you will notice that mixed-precision doesn't always reduce memory usage (e.g. full training step, context length 128, 32G -> 36G). Those can happen when precision mismatch triggers implicit casts which are temporary copies and increases memory footprint, or when there's autograd overhead caused by mixed-precision.

GPT's answer: https://chatgpt.com/share/688fd6f9-ab54-8013-83a4-7a85e6f4f5cf 

(d)  For GPT-XL, context length 256, batch size of 4, d_model = 1600, that's 256 * 4 * 1600 * 8 =13,107,200  bytes, so that's 12.5MB.

For 2.7B, context length 256, batch size of 4, d_model = 2560, that's 256 * 4 * 2560 * 8 =  2,0971,520 bytes, so that's 20MB.

(e) When reducing details to a small number, the largest allocation is of size 100MB (104,857,600 bytes). The stacktrace looks like the allocation is done in softmax in `nn_utils.py`, which invoked `torch.exp`.

The 100MB is probably because PyTorch allocates memory in fixed sizes. 

## Problem (pytorch_attention)

Benchmarked on H100 with 80G RAM (100 passes).

| d_model | seq_len | forward time | CUDA memory usage after forward | backward time |
| ------: | ------: | -----------: | ------------------------------: | ------------: |
|      16 |     256 |       0.0312 |                         68.9614 |        0.0898 |
|      16 |    1024 |       0.0476 |                         132.594 |        0.0937 |
|      16 |    4096 |       0.3129 |                         1118.38 |        0.6993 |
|      16 |    8192 |       1.2175 |                         4252.75 |        2.6391 |
|      16 |   16384 |       4.7718 |                         16761.5 |       10.3709 |
|      32 |     256 |       0.0272 |                         69.8364 |        0.0819 |
|      32 |    1024 |       0.0418 |                         136.094 |        0.0895 |
|      32 |    4096 |       0.3258 |                         1132.38 |        0.7228 |
|      32 |    8192 |       1.2485 |                         4280.75 |        2.6671 |
|      32 |   16384 |       4.8859 |                         16817.5 |       10.4795 |
|      64 |     256 |         0.03 |                         71.5864 |        0.0814 |
|      64 |    1024 |       0.0429 |                         143.094 |        0.0857 |
|      64 |    4096 |       0.3494 |                         1160.38 |         0.766 |
|      64 |    8192 |       1.3693 |                         4336.75 |        2.9065 |
|      64 |   16384 |       5.4185 |                         16929.5 |       11.4581 |
|     128 |     256 |       0.0271 |                         75.0864 |         0.082 |
|     128 |    1024 |       0.0434 |                         157.094 |        0.0941 |
|     128 |    4096 |       0.4119 |                         1216.38 |         0.898 |
|     128 |    8192 |       1.6008 |                         4448.75 |        3.3566 |
|     128 |   16384 |       6.3116 |                         17153.5 |       13.2443 |

When increasing seq_len from 256 to 8192 (32x), time increases for ~80x,  the memory increases for ~60x. 

When increasing seq_len from 256 to 16382 (64x), time increases for ~315x, the memory increases for ~240x.

Observation: Superlinear scaling. Long sequences are very expensive.

## Problem (torch_compile)

Benchmarked on H100 with 80G RAM.

Attention benchmark 

| d_model | seq_len | baseline forward | compile forward | compile w/ float32 forward | baseline backward | compile backward | compile w/ float32 backward | CUDA memory usage after forward |
| ------: | ------: | ---------------: | --------------: | -------------------------: | ----------------: | ---------------: | --------------------------: | ------------------------------: |
|      16 |     256 |           0.0312 |           0.036 |                     0.0176 |            0.0898 |           0.0539 |                      0.0544 |                         68.9614 |
|      16 |    1024 |           0.0476 |          0.0418 |                     0.0213 |            0.0937 |           0.0566 |                      0.0514 |                         132.594 |
|      16 |    4096 |           0.3129 |          0.1277 |                     0.1114 |            0.6993 |           0.2942 |                      0.2738 |                         1118.38 |
|      16 |    8192 |           1.2175 |          0.5273 |                     0.4749 |            2.6391 |           1.0851 |                      0.9671 |                         4252.75 |
|      16 |   16384 |           4.7718 |          1.8686 |                     1.6892 |           10.3709 |           4.2687 |                      3.8779 |                         16761.5 |
|      32 |     256 |           0.0272 |          0.0491 |                     0.0263 |            0.0819 |           0.0561 |                      0.0517 |                         69.8364 |
|      32 |    1024 |           0.0418 |          0.0396 |                     0.0371 |            0.0895 |           0.0558 |                      0.0576 |                         136.094 |
|      32 |    4096 |           0.3258 |          0.1728 |                     0.1532 |            0.7228 |           0.3258 |                      0.2893 |                         1132.38 |
|      32 |    8192 |           1.2485 |          0.6303 |                     0.5542 |            2.6671 |           1.1574 |                      1.0147 |                         4280.75 |
|      32 |   16384 |           4.8859 |          2.1536 |                     1.8669 |           10.4795 |           4.4769 |                      3.9683 |                         16817.5 |
|      64 |     256 |             0.03 |          0.0265 |                     0.0236 |            0.0814 |           0.0559 |                      0.0521 |                         71.5864 |
|      64 |    1024 |           0.0429 |          0.0428 |                      0.038 |            0.0857 |           0.0582 |                      0.0586 |                         143.094 |
|      64 |    4096 |           0.3494 |          0.1941 |                     0.1474 |             0.766 |           0.3777 |                      0.2915 |                         1160.38 |
|      64 |    8192 |           1.3693 |          0.6771 |                     0.4756 |            2.9065 |           1.3774 |                      1.0033 |                         4336.75 |
|      64 |   16384 |           5.4185 |          2.6819 |                     1.8678 |           11.4581 |           5.4604 |                      3.9668 |                         16929.5 |
|     128 |     256 |           0.0271 |          0.0261 |                     0.0233 |             0.082 |           0.0568 |                      0.0515 |                         75.0864 |
|     128 |    1024 |           0.0434 |          0.0411 |                     0.0392 |            0.0941 |           0.0676 |                      0.0616 |                         157.094 |
|     128 |    4096 |           0.4119 |          0.2556 |                     0.1503 |             0.898 |           0.5002 |                         0.3 |                         1216.38 |
|     128 |    8192 |           1.6008 |          0.9092 |                     0.4949 |            3.3566 |           1.8287 |                      1.0355 |                         4448.75 |
|     128 |   16384 |           6.3116 |          3.5884 |                     1.9464 |           13.2443 |           7.2555 |                      4.0837 |                         17153.5 |

We observed similar superlinear scaling for sequence length. For example, for d_model=128, when seq_len increases from 256 to 8192 (32x), time increases ~20x, RAM increases ~60x; when seq_len increases from 256 to 16382 (64x), time increases ~80x, RAM increases ~240x.  

Transformer benchmark (forward + backward + optimizer)

| d_model |  d_ff | num_layers | num_heads | context_length | mean_time (baseline) | mean_time (compile) | mean_time (compile w/ f32)                      |
| ------: | ----: | ---------: | --------: | -------------: | -------------------: | ------------------: | ----------------------------------------------: |
|     768 |  3072 |         12 |        12 |            128 |             0.073000 |            0.049800 |                                        0.052900 |
|     768 |  3072 |         12 |        12 |            256 |             0.074300 |            0.051300 |                                        0.053500 |
|     768 |  3072 |         12 |        12 |            512 |             0.082700 |            0.056600 |                                        0.050800 |
|    1024 |  4096 |         24 |        16 |            128 |             0.145300 |            0.076100 |                                        0.077400 |
|    1024 |  4096 |         24 |        16 |            256 |             0.146500 |            0.097000 |                                        0.088100 |
|    1024 |  4096 |         24 |        16 |            512 |             0.210100 |            0.159000 |                                        0.084300 |
|    1280 |  5120 |         36 |        20 |            128 |             0.212200 |            0.156400 |                                        0.133600 |
|    1280 |  5120 |         36 |        20 |            256 |             0.267600 |            0.226400 |                                        0.133500 |
|    1280 |  5120 |         36 |        20 |            512 |             0.443800 |            0.375800 |                                        0.162800 |
|    1600 |  6400 |         48 |        25 |            128 |             0.358100 |            0.295100 |                                        0.207200 |
|    1600 |  6400 |         48 |        25 |            256 |             0.501500 |            0.452000 |                                        0.231400 |
|    1600 |  6400 |         48 |        25 |            512 |             0.858800 |            0.749400 |                                        0.283500 |
|    2560 | 10240 |         32 |        32 |            128 |             0.471300 |            0.446800 |                                        0.270300 |
|    2560 | 10240 |         32 |        32 |            256 |             0.724500 |            0.680600 |                                        0.303200 |
|    2560 | 10240 |         32 |        32 |            512 |             1.246300 |            1.143300 |                                        0.393300 |

We observed similar superlinear scaling. For example, for d_model = 2560, when context_length increases from 128 to 256 (2x), the time increases by ~1.122x; when context_length increases from 128 to 512 (4x), the time increases by 1.455x. 1.122^2 =1.259 < 1.455x. 

## Problem (flash_forward)

(a) I initially wrote a bug to flatten all batch dimensions, mixing attention across batches. This is wrong because 

* Attention produces how one sequence (that provides Q) should attend to another sequence (that provides K, V).
* In the case of self-attention, each batch is a distinct input sequence. E.g. batch 1: "The cat sat on the mat", batch 2 "How are you?".  Mixing the batch is meaningless (e.g. "The cat sat on the mat How are you?").
* The softmax is computed over the sequence to attend to. Mixing batches extends the sequence in a meaningless way.
* The batch dimension serves as a hard boundary that preserves the boundary of different input sequences.

Claude found the bug for me. See Appendix for Claude's debugging process.

## Problem (flash_benchmarking)

For float32

| Seq Len | D Model | Dtype   | Q Tile | K Tile | Best | Fwd PT (ms) | Fwd TR (ms) | Fwd Speedup | Bwd PT (ms) | Bwd TR (ms) | Bwd Speedup | E2E PT (ms) | E2E TR (ms) | E2E Speedup |
| ------: | ------: | :------ | -----: | -----: | :--- | ----------: | ----------: | :---------- | ----------: | ----------: | :---------- | ----------: | ----------: | :---------- |
|     128 |      16 | float32 |     32 |     64 | ✓    |       0.094 |       0.009 | 10.81x      |       0.787 |        2.39 | 0.33x       |       0.881 |       2.398 | 0.37x       |
|     128 |      32 | float32 |     32 |     64 | ✓    |       0.143 |       0.009 | 15.47x      |       0.901 |       2.302 | 0.39x       |       1.044 |       2.311 | 0.45x       |
|     128 |      64 | float32 |     16 |     32 | ✓    |       0.205 |        0.01 | 20.06x      |       1.019 |        2.51 | 0.41x       |       1.224 |        2.52 | 0.49x       |
|     128 |     128 | float32 |     16 |     32 | ✓    |       0.132 |       0.013 | 10.47x      |       1.035 |       2.632 | 0.39x       |       1.167 |       2.645 | 0.44x       |
|     256 |      16 | float32 |     16 |     16 | ✓    |       0.167 |       0.019 | 8.99x       |       1.269 |        1.75 | 0.73x       |       1.436 |       1.768 | 0.81x       |
|     256 |      32 | float32 |     32 |     64 | ✓    |       0.238 |       0.025 | 9.58x       |       1.201 |       2.227 | 0.54x       |       1.439 |       2.252 | 0.64x       |
|     256 |      64 | float32 |     16 |     16 | ✓    |       0.182 |        0.02 | 8.95x       |       1.274 |       3.051 | 0.42x       |       1.456 |       3.072 | 0.47x       |
|     256 |     128 | float32 |     16 |     32 | ✓    |       0.212 |       0.029 | 7.42x       |       1.186 |         2.7 | 0.44x       |       1.398 |       2.728 | 0.51x       |
|     512 |      16 | float32 |     32 |    128 | ✓    |       0.212 |       0.014 | 14.80x      |       1.198 |       2.792 | 0.43x       |        1.41 |       2.806 | 0.50x       |
|     512 |      32 | float32 |     32 |    128 | ✓    |        0.23 |       0.017 | 13.78x      |       1.256 |       2.634 | 0.48x       |       1.487 |        2.65 | 0.56x       |
|     512 |      64 | float32 |     32 |    128 | ✓    |       0.206 |       0.021 | 9.95x       |       1.208 |        2.36 | 0.51x       |       1.414 |        2.38 | 0.59x       |
|    1024 |      16 | float32 |     32 |    128 | ✓    |       0.205 |       0.032 | 6.39x       |       1.206 |        2.69 | 0.45x       |       1.411 |       2.722 | 0.52x       |
|    1024 |      32 | float32 |     32 |     64 | ✓    |       0.208 |       0.035 | 5.94x       |       1.216 |       2.561 | 0.47x       |       1.423 |       2.596 | 0.55x       |
|    1024 |      64 | float32 |     32 |    128 | ✓    |       0.208 |       0.042 | 4.94x       |       1.207 |       2.695 | 0.45x       |       1.415 |       2.737 | 0.52x       |
|    2048 |      16 | float32 |     32 |    128 | ✓    |       0.208 |       0.044 | 4.76x       |       1.212 |        2.55 | 0.48x       |        1.42 |       2.593 | 0.55x       |
|    2048 |      32 | float32 |     32 |     64 | ✓    |       0.225 |        0.05 | 4.50x       |       1.232 |       2.705 | 0.46x       |       1.457 |       2.755 | 0.53x       |
|    2048 |      64 | float32 |     32 |    128 | ✓    |       0.212 |       0.064 | 3.30x       |       1.231 |       2.534 | 0.49x       |       1.444 |       2.598 | 0.56x       |
|    4096 |      16 | float32 |     32 |    128 | ✓    |       0.219 |       0.069 | 3.17x       |       1.191 |       2.731 | 0.44x       |       1.409 |         2.8 | 0.50x       |
|    4096 |      32 | float32 |     32 |    128 | ✓    |       0.217 |        0.08 | 2.70x       |       1.238 |       2.329 | 0.53x       |       1.455 |       2.409 | 0.60x       |
|    4096 |      64 | float32 |     32 |    128 | ✓    |       0.212 |       0.115 | 1.85x       |       1.004 |       2.579 | 0.39x       |       1.217 |       2.694 | 0.45x       |
|    8192 |      16 | float32 |     64 |    128 | ✓    |       0.475 |       0.173 | 2.75x       |       1.129 |       2.726 | 0.41x       |       1.604 |       2.899 | 0.55x       |
|    8192 |      32 | float32 |     32 |    128 | ✓    |       0.476 |        0.24 | 1.98x       |       1.119 |        2.46 | 0.45x       |       1.596 |       2.701 | 0.59x       |
|    8192 |      64 | float32 |     32 |     64 | ✓    |       0.487 |       0.335 | 1.45x       |       0.762 |       1.363 | 0.56x       |       1.249 |       1.698 | 0.74x       |
|   16384 |      16 | float32 |     64 |     64 | ✓    |       1.963 |       0.402 | 4.89x       |       3.171 |        3.39 | 0.94x       |       5.135 |       3.792 | 1.35x       |
|   16384 |      32 | float32 |     64 |     64 | ✓    |       1.979 |       0.572 | 3.46x       |       3.208 |       3.434 | 0.93x       |       5.187 |       4.005 | 1.30x       |
|   16384 |      64 | float32 |     32 |     64 | ✓    |        2.01 |       1.292 | 1.56x       |       3.258 |       3.525 | 0.92x       |       5.268 |       4.816 | 1.09x       |
|   32768 |      16 | float32 |    128 |     64 | ✓    |       9.638 |       1.345 | 7.17x       |        12.9 |      13.285 | 0.97x       |      22.539 |       14.63 | 1.54x       |
|   32768 |      32 | float32 |     64 |     32 | ✓    |       9.634 |       2.175 | 4.43x       |      12.895 |        13.3 | 0.97x       |      22.528 |      15.475 | 1.46x       |
|   32768 |      64 | float32 |    128 |     32 | ✓    |      10.097 |       4.214 | 2.40x       |      13.504 |      14.011 | 0.96x       |        23.6 |      18.225 | 1.29x       |
|   65536 |      16 | float32 |    128 |     64 | ✓    |      38.862 |       5.632 | 6.90x       |       52.35 |      58.471 | 0.90x       |      91.213 |      64.102 | 1.42x       |
|   65536 |      32 | float32 |     64 |     32 | ✓    |      39.239 |       8.671 | 4.53x       |      52.844 |      59.789 | 0.88x       |      92.083 |       68.46 | 1.35x       |
|   65536 |      64 | float32 |    128 |     32 | ✓    |      39.567 |      18.299 | 2.16x       |      52.878 |      60.911 | 0.87x       |      92.444 |       79.21 | 1.17x       |

For bfloat16

| Seq Len | D Model | Dtype    | Q Tile | K Tile | Best | Fwd PT (ms) | Fwd TR (ms) | Fwd Speedup | Bwd PT (ms) | Bwd TR (ms) | Bwd Speedup | E2E PT (ms) | E2E TR (ms) | E2E Speedup |
| ------: | ------: | :------- | -----: | -----: | :--- | ----------: | ----------: | :---------- | ----------: | ----------: | :---------- | ----------: | ----------: | :---------- |
|     128 |      16 | bfloat16 |     32 |     64 | ✓    |       0.095 |       0.008 | 11.31x      |       1.161 |         2.4 | 0.48x       |       1.257 |       2.408 | 0.52x       |
|     128 |      32 | bfloat16 |     16 |     32 | ✓    |       0.147 |       0.009 | 15.75x      |       1.249 |       1.723 | 0.72x       |       1.396 |       1.733 | 0.81x       |
|     128 |      64 | bfloat16 |     16 |     32 | ✓    |       0.149 |       0.009 | 15.94x      |       0.911 |       2.597 | 0.35x       |        1.06 |       2.607 | 0.41x       |
|     128 |     128 | bfloat16 |     16 |     32 | ✓    |       0.134 |       0.011 | 12.60x      |       1.025 |       2.469 | 0.41x       |       1.158 |        2.48 | 0.47x       |
|     256 |      16 | bfloat16 |     16 |     16 | ✓    |        0.17 |       0.014 | 11.96x      |       0.919 |       1.775 | 0.52x       |        1.09 |       1.789 | 0.61x       |
|     256 |      32 | bfloat16 |     16 |     16 | ✓    |        0.21 |       0.018 | 11.48x      |       1.191 |       2.565 | 0.46x       |         1.4 |       2.583 | 0.54x       |
|     256 |      64 | bfloat16 |     16 |     16 | ✓    |         0.2 |       0.018 | 10.88x      |       1.219 |       2.829 | 0.43x       |       1.419 |       2.847 | 0.50x       |
|     256 |     128 | bfloat16 |     16 |     16 | ✓    |       0.222 |       0.024 | 9.23x       |       1.188 |        2.53 | 0.47x       |        1.41 |       2.554 | 0.55x       |
|     512 |      16 | bfloat16 |     32 |    128 | ✓    |         0.2 |       0.013 | 14.98x      |       1.188 |       2.604 | 0.46x       |       1.388 |       2.617 | 0.53x       |
|     512 |      32 | bfloat16 |     32 |    128 | ✓    |       0.197 |       0.015 | 13.31x      |       1.195 |       2.817 | 0.42x       |       1.392 |       2.832 | 0.49x       |
|     512 |      64 | bfloat16 |     32 |    128 | ✓    |        0.16 |       0.016 | 10.14x      |       1.036 |       2.849 | 0.36x       |       1.196 |       2.865 | 0.42x       |
|     512 |     128 | bfloat16 |     32 |    128 | ✓    |       0.163 |        0.02 | 8.10x       |       1.107 |       2.755 | 0.40x       |        1.27 |       2.775 | 0.46x       |
|    1024 |      16 | bfloat16 |     32 |    128 | ✓    |       0.212 |       0.031 | 6.82x       |       1.207 |       2.594 | 0.47x       |       1.419 |       2.625 | 0.54x       |
|    1024 |      32 | bfloat16 |     32 |    128 | ✓    |       0.208 |       0.033 | 6.30x       |       1.222 |       2.577 | 0.47x       |       1.429 |        2.61 | 0.55x       |
|    1024 |      64 | bfloat16 |     32 |    128 | ✓    |       0.212 |       0.034 | 6.27x       |       1.205 |       2.583 | 0.47x       |       1.417 |       2.617 | 0.54x       |
|    1024 |     128 | bfloat16 |     32 |    128 | ✓    |       0.202 |        0.04 | 5.09x       |       1.245 |       2.586 | 0.48x       |       1.447 |       2.626 | 0.55x       |
|    2048 |      16 | bfloat16 |     32 |    128 | ✓    |       0.183 |       0.048 | 3.79x       |        0.95 |       2.811 | 0.34x       |       1.133 |       2.859 | 0.40x       |
|    2048 |      32 | bfloat16 |     64 |    128 | ✓    |       0.215 |       0.042 | 5.13x       |       1.222 |       2.841 | 0.43x       |       1.438 |       2.883 | 0.50x       |
|    2048 |      64 | bfloat16 |     64 |    128 | ✓    |       0.247 |       0.046 | 5.41x       |       1.239 |       2.426 | 0.51x       |       1.486 |       2.472 | 0.60x       |
|    2048 |     128 | bfloat16 |     32 |    128 | ✓    |        0.23 |       0.062 | 3.70x       |       1.258 |         2.5 | 0.50x       |       1.488 |       2.562 | 0.58x       |
|    4096 |      16 | bfloat16 |     32 |    128 | ✓    |       0.213 |       0.066 | 3.21x       |       1.373 |       2.669 | 0.51x       |       1.586 |       2.735 | 0.58x       |
|    4096 |      32 | bfloat16 |     32 |    128 | ✓    |       0.216 |       0.073 | 2.95x       |       1.011 |       2.455 | 0.41x       |       1.226 |       2.528 | 0.49x       |
|    4096 |      64 | bfloat16 |     32 |    128 | ✓    |       0.259 |       0.077 | 3.37x       |       1.342 |       2.629 | 0.51x       |       1.601 |       2.706 | 0.59x       |
|    4096 |     128 | bfloat16 |     32 |    128 | ✓    |       0.211 |        0.11 | 1.93x       |       1.265 |        2.58 | 0.49x       |       1.476 |        2.69 | 0.55x       |
|    8192 |      16 | bfloat16 |     64 |     64 | ✓    |       0.469 |       0.143 | 3.28x       |       1.008 |       2.861 | 0.35x       |       1.476 |       3.004 | 0.49x       |
|    8192 |      32 | bfloat16 |     64 |    128 | ✓    |       0.462 |       0.149 | 3.11x       |       1.258 |       2.647 | 0.48x       |       1.721 |       2.796 | 0.62x       |
|    8192 |      64 | bfloat16 |     64 |    128 | ✓    |       0.469 |       0.159 | 2.95x       |       1.216 |       2.619 | 0.46x       |       1.685 |       2.778 | 0.61x       |
|   16384 |      16 | bfloat16 |     64 |     64 | ✓    |       1.767 |        0.37 | 4.77x       |       2.225 |       1.857 | 1.20x       |       3.992 |       2.227 | 1.79x       |
|   16384 |      32 | bfloat16 |     64 |    128 | ✓    |        1.76 |       0.365 | 4.83x       |       2.258 |       2.727 | 0.83x       |       4.018 |       3.091 | 1.30x       |
|   16384 |      64 | bfloat16 |     64 |    128 | ✓    |       1.766 |       0.409 | 4.32x       |       2.264 |        1.88 | 1.20x       |        4.03 |       2.289 | 1.76x       |
|   32768 |      16 | bfloat16 |     64 |     64 | ✓    |       6.982 |       1.223 | 5.71x       |        8.69 |       7.171 | 1.21x       |      15.672 |       8.394 | 1.87x       |
|   32768 |      32 | bfloat16 |     64 |     64 | ✓    |       6.951 |       1.303 | 5.33x       |       8.771 |       7.157 | 1.23x       |      15.722 |       8.461 | 1.86x       |
|   32768 |      64 | bfloat16 |     64 |    128 | ✓    |        6.99 |       1.597 | 4.38x       |        8.81 |        7.25 | 1.22x       |      15.801 |       8.847 | 1.79x       |
|   65536 |      16 | bfloat16 |     64 |     64 | ✓    |      29.111 |       4.738 | 6.14x       |      38.416 |      43.994 | 0.87x       |      67.526 |      48.733 | 1.39x       |
|   65536 |      32 | bfloat16 |     64 |    128 | ✓    |      29.084 |       4.982 | 5.84x       |      38.339 |      43.986 | 0.87x       |      67.423 |      48.968 | 1.38x       |
|   65536 |      64 | bfloat16 |     64 |    128 | ✓    |        29.3 |       6.297 | 4.65x       |      39.078 |      44.382 | 0.88x       |      68.378 |      50.679 | 1.35x       |

Observations:

* Significant speedup in the forward pass. The backward pass is slower than compiled PyTorch implementation, because we are not using Triton.
* The speedup for forward pass dimishes for longer sequences, probably because we become more compute-bound when the sequence length increases.
* For forward, only see noticable speedup from bfloat16 for long sequence length (32768). This also makes sense because there's higher memory pressure for longer sequences.

## Problem (distributed_communication_single_node)

Note: benchmarked on CPU.

|        | world size = 2 | world size = 4 | world size = 6 |
| ------ | -------------- | -------------- | -------------- |
| 1MB    | 0.024          | 0.029          | 0.06           |
| 10MB   | 0.061          | 0.14           | 0.26           |
| 100MB  | 0.62           | 1.42           | 2.95           |
| 1000MB | 6.25           | 15.7           | 34.6           |

With 2 processes, we get linear scaling. At 1MB, the time is likely dominated by overhead (compare with world size of 4).

With 4 processes, we also get almost linear scaling. 

With 6 processes, we get superlinear scaling w.r.t data size. Probably because of more significant synchronization overhead (e.g. faster worker waiting for slower ones)?





## Appendix

## Debugging non-Triton FlashAttention

Claude debugging my non-Triton FlashAttention kernel implementation. Note that point 3 is wrong, that's not an actual bug. It's still impressive that Claude was able to debug in such a methodological and rigorous approach.

```
Here's a systematic breakdown of the debugging process and hypotheses I tested:

    1. Initial Algorithm Correctness Check

  - Hypothesis: The tiling algorithm implementation doesn't match FlashAttention-2 spec
  - Test: Compared implementation line-by-line against the provided algorithm
  - Result: Found minor issues (initialization, normalization) but algorithm structure
    was correct

    2. Numerical Precision Issues

  - Hypothesis: Small floating-point differences causing test failure
  - Test: Created simple test cases with known inputs, measured differences
  - Result: ❌ Differences were massive (92.4% mismatch), not precision issues

    3. Tensor Initialization Bug

  - Hypothesis: Using tensor views instead of independent copies corrupts computation
  - Test: Changed Oij = O[i * Bq : (i + 1) * Bq, :] to Oij = torch.zeros((Bq, d), 
    device=Q.device)
  - Result: ✅ This was a real bug - using views meant modifying final output during
    computation

    4. Device Specification Issues

  - Hypothesis: Mixed CPU/GPU tensors causing computation errors
  - Test: Added device=Q.device to all tensor creations including mij
  - Result: ✅ Minor fix, but necessary for GPU compatibility

    5. Single Tile vs Multi-Tile Testing

  - Hypothesis: Multi-tile logic has bugs in tile boundary handling
  - Test: Created step-by-step debug of first tile computation, then multi-tile cases
  - Result: ✅ Single tile worked perfectly, multi-tile also worked for small cases

    6. Dimension Scaling Analysis

  - Hypothesis: Algorithm breaks at certain dimensions due to numerical instability
  - Test: Gradually scaled from 8x16x4 to 128x128x64 dimensions
  - Result: ❌ All dimension scales worked for single batch

    7. Batch Processing Investigation (The Root Cause)

  - Hypothesis: The batch dimension flattening is mixing data across batches
  - Test: Compared single batch vs multi-batch behavior with identical per-batch data
  - Result: ✅ CRITICAL BUG FOUND - rearrange(Q, "... d -> (...) d") was flattening ALL
    dimensions, causing tiles to contain mixed batch data

    8. Final Validation

  - Test: Rewrote to process each batch independently while maintaining tiling within
    batches
  - Result: ✅ All tests pass, including the original failing test

  Key Insights:

    1. Algorithm was mathematically correct - the issue was in data organization
    2. Batch boundaries matter - attention must be computed independently per batch
    3. Systematic scaling testing was crucial to isolate the batch size as the breaking
       point
    4. Step-by-step debugging of intermediate computations revealed the exact point of
       failure

  The debugging process took a methodical approach: algorithm → precision → implementation details → data organization, ultimately finding that the core mathematical implementation was sound but the batch handling was fundamentally flawed.
```







