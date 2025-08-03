## problem(benchmarking_script)

(a) `benchmark.py` and `benchmark_sweep.py`. See results in `data/e2e_benchmark`. 

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