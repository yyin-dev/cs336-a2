# CS336 Systems: GPU Training Performance

This is my implementation and performance study for Stanford CS336 Assignment 2, [Systems](https://github.com/stanford-cs336/assignment2-systems/blob/main/cs336_assignment2_systems.pdf).

The goal is to make Transformer training faster on one GPU and scale it across multiple GPUs by building the core systems pieces directly: profiling instrumentation, mixed precision, FlashAttention, distributed data parallelism (DDP), communication overlap, and optimizer-state sharding. See details in [writeup](writeup.md).

## Highlights

- Implemented a Transformer training benchmark suite with NVTX ranges and Nsight Systems traces for forward, backward, and optimizer phases.
- Profiled GPT-style models on A100 40GB and H100 80GB GPUs, including GPT-2 small through GPT-2 XL sized configurations.
- Measured full training-step performance and identified the workload shift from matmul-dominated inference/forward passes to elementwise-heavy backward and optimizer steps.
- Benched mixed-precision training. On GPT-2 XL with context length 256, a full forward/backward/optimizer step improved from about `1.09s` to `0.44s`.
- Implemented FlashAttention-style online softmax in PyTorch and Triton and ran performance benchmark. Benchmarked Triton FlashAttention over sequence lengths up to `65,536`, measuring forward, backward, and end-to-end behavior in `float32` and `bfloat16`.
- **Implemented and benchmarked several DDP variants**: naive per-parameter all-reduce, flattened-gradient sync, per-parameter communication overlap, and bucketed communication overlap. With 2-GPU setup, on GPT-Large with sequence length 64, communication overlap reduced step time from `3.44s` to `3.26s`; on GPT-XL, overlap reduced `5.045s` to `4.359s`.
- **Implemented optimizer-state sharding for AdamW-style optimizers for multi-GPU training**. With world size 2, peak memory dropped by about `8.9GB`, matching the expected savings from sharding half of the optimizer state.

## Selected Results

| Area | Result |
| --- | --- |
| Kernel profile | GPT-2 XL forward pass spent over 85% of time in matmul kernels; full training step dropped to about 68% matmul because backward and optimizer add many elementwise kernels. |
| Mixed precision | GPT-2 XL, context 256, batch 4 improved from `1.09s` to `0.44s` per full training step. |
| Attention scaling | Standard attention showed superlinear time and memory growth with sequence length, making long-context attention the central bottleneck. |
| Triton FlashAttention | Forward pass was substantially faster than eager PyTorch attention across many regimes; end-to-end gains were limited by the backward implementation. |
| DDP overlap | Per-parameter overlap and small buckets exposed less communication time than naive or fully flattened synchronization on the tested 2-GPU setup. |
| Optimizer sharding | World-size-2 sharding saved about one model-sized copy of AdamW state per worker, at about `8%` step-time overhead in the benchmark. |

## Implementation Notes

The DDP wrappers use `torch.distributed` directly so the communication behavior is visible:

- `DDPNaive` all-reduces each gradient after backward.
- `DDPBatch` flattens gradients into one all-reduce to reduce launch overhead.
- `DDPOverlapIndividualParams` starts asynchronous all-reduce from gradient hooks as soon as each parameter gradient is produced.
- `DDPOverlapBucketed` groups parameters into reverse-order buckets, launches asynchronous all-reduce as each bucket becomes ready, and unflattens the synchronized gradients before the optimizer step.

The optimizer sharding wrapper stores the full parameter set on each worker for forward/backward, but
only assigns each rank a subset of optimizer states. After each local optimizer step, updated parameter
shards are broadcast so every worker returns to a full, synchronized model.

The FlashAttention implementation uses online softmax to avoid materializing the full attention matrix
in the forward pass. The tests compare saved log-sum-exp values, outputs, and gradients against a
standard PyTorch attention reference, including causal attention on CUDA.

## Directory Structure

```
cs336_systems/
  flashattention.py              # PyTorch + Triton FlashAttention implementations
  ddp_naive.py                   # baseline DDP wrapper
  ddp_batch.py                   # flattened-gradient all-reduce
  ddp_overlap_individual_params.py
  ddp_overlap_bucketed.py        # bucketed overlap of backward compute and all-reduce
  optimizer_state_sharding.py    # sharded optimizer-state wrapper
  model.py, nn_utils.py          # Transformer/model utilities used in benchmarks

benchmark/                       # Benchmark scripts
```
