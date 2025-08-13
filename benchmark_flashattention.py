"""
Benchmark PyTorch attention vs. Triton FlashAttention.
"""

import torch
import triton.testing
import itertools
from einops import einsum, rearrange
import math
import pandas as pd
from flashattention import FlashAttentionTriton


def attention(Q, K, V, is_causal=False):
    """
    Args
        Q: ... m d
        K: ... n d
        V: ... n d
    """

    m = Q.shape[-2]
    n = K.shape[-2]
    d = Q.shape[-1]

    S = einsum(Q, K, "... m d, ... n d -> ... m n") / math.sqrt(d)
    if is_causal:
        q_indices = torch.arange(0, m, device=Q.device)
        k_indices = torch.arange(0, n, device=Q.device)
        qs = rearrange(q_indices, "(m x) -> m x", x=1)
        ks = rearrange(k_indices, "(n x) -> x n", x=1)
        mask = qs >= ks
        S = torch.where(mask, S, float("-inf"))

    P = torch.softmax(S, dim=-1)
    O = einsum(P, V, "... m n, ... n d -> ... m d")

    return O


def generate_inputs(batch_size, seq_len, d_model, dtype, device="cuda"):
    """Generate random inputs for benchmarking"""
    torch.manual_seed(42)  # For reproducibility

    Q = torch.randn(
        batch_size, seq_len, d_model, dtype=dtype, device=device, requires_grad=True
    )
    K = torch.randn(
        batch_size, seq_len, d_model, dtype=dtype, device=device, requires_grad=True
    )
    V = torch.randn(
        batch_size, seq_len, d_model, dtype=dtype, device=device, requires_grad=True
    )

    return Q, K, V


def benchmark_forward_pytorch(Q, K, V, is_causal=True):
    """Benchmark PyTorch forward pass"""
    compiled_attention = torch.compile(attention)

    def fn():
        return compiled_attention(Q, K, V, is_causal)

    return triton.testing.do_bench(fn)


def benchmark_forward_triton(Q, K, V, is_causal=True):
    """Benchmark Triton forward pass"""

    def fn():
        return FlashAttentionTriton.apply(Q, K, V, is_causal)

    return triton.testing.do_bench(fn)


def benchmark_end_to_end_pytorch(Q, K, V, is_causal=True):
    """Benchmark PyTorch end-to-end forward + backward pass"""
    compiled_attention = torch.compile(attention)

    def fn():
        Q_clone = Q.clone().detach().requires_grad_(True)
        K_clone = K.clone().detach().requires_grad_(True)
        V_clone = V.clone().detach().requires_grad_(True)
        O = compiled_attention(Q_clone, K_clone, V_clone, is_causal)
        loss = O.sum()
        loss.backward()

    return triton.testing.do_bench(fn)


def benchmark_end_to_end_triton(Q, K, V, is_causal=True):
    """Benchmark Triton end-to-end forward + backward pass"""

    def fn():
        Q_clone = Q.clone().detach().requires_grad_(True)
        K_clone = K.clone().detach().requires_grad_(True)
        V_clone = V.clone().detach().requires_grad_(True)
        O = FlashAttentionTriton.apply(Q_clone, K_clone, V_clone, is_causal)
        loss = O.sum()
        loss.backward()

    return triton.testing.do_bench(fn)


def run_benchmark():
    """Run comprehensive benchmark comparing PyTorch and Triton implementations"""

    # Check device
    if not torch.cuda.is_available():
        print("CUDA not available. This benchmark requires a GPU.")
        return

    torch.set_float32_matmul_precision("high")
    device = "cuda"
    batch_size = 1
    is_causal = True

    # Parameter ranges - start with smaller subset for testing
    seq_lengths = [2**i for i in range(7, 17)]  # 128 to 65536
    d_models = [2**i for i in range(4, 8)]  # 16 to 128
    dtypes = [torch.float32]  # TODO: include torch.bfloat16

    print("FlashAttention Benchmark Results")
    print("=" * 80)
    print(f"Batch Size: {batch_size}, Causal Masking: {is_causal}")
    print("=" * 80)

    results = []

    for seq_len, d_model, dtype in itertools.product(seq_lengths, d_models, dtypes):
        try:
            # Generate inputs
            Q, K, V = generate_inputs(batch_size, seq_len, d_model, dtype, device)

            # Warm up GPU
            for _ in range(3):
                _ = attention(Q, K, V, is_causal)
                _ = FlashAttentionTriton.apply(Q, K, V, is_causal)

            torch.cuda.synchronize()

            # Benchmark forward pass
            fwd_pytorch_time = benchmark_forward_pytorch(Q, K, V, is_causal)
            fwd_triton_time = benchmark_forward_triton(Q, K, V, is_causal)

            # Benchmark end-to-end (forward + backward)
            e2e_pytorch_time = benchmark_end_to_end_pytorch(Q, K, V, is_causal)
            e2e_triton_time = benchmark_end_to_end_triton(Q, K, V, is_causal)

            # Calculate backward time by subtraction
            bwd_pytorch_time = e2e_pytorch_time - fwd_pytorch_time
            bwd_triton_time = e2e_triton_time - fwd_triton_time

            results.append(
                {
                    "seq_len": seq_len,
                    "d_model": d_model,
                    "dtype": dtype,
                    "fwd_pytorch": fwd_pytorch_time,
                    "fwd_triton": fwd_triton_time,
                    "bwd_pytorch": bwd_pytorch_time,
                    "bwd_triton": bwd_triton_time,
                    "e2e_pytorch": e2e_pytorch_time,
                    "e2e_triton": e2e_triton_time,
                }
            )

            print(
                f"Seq Len: {seq_len:5d}, D Model: {d_model:3d}, "
                f"Dtype: {str(dtype).split('.')[-1]:8s} - "
                f"Fwd: PT {fwd_pytorch_time:.3f}ms / TR {fwd_triton_time:.3f}ms, "
                f"Bwd: PT {bwd_pytorch_time:.3f}ms / TR {bwd_triton_time:.3f}ms, "
                f"E2E: PT {e2e_pytorch_time:.3f}ms / TR {e2e_triton_time:.3f}ms"
            )

        except Exception as e:
            print(
                f"Error with seq_len={seq_len}, d_model={d_model}, dtype={dtype}: {e}"
            )
            continue

    # Create DataFrame for better formatting
    df_data = []
    for result in results:
        fwd_speedup = result["fwd_pytorch"] / result["fwd_triton"]
        bwd_speedup = result["bwd_pytorch"] / result["bwd_triton"]
        e2e_speedup = result["e2e_pytorch"] / result["e2e_triton"]

        dtype_str = str(result["dtype"]).split(".")[-1]

        df_data.append(
            {
                "Seq Len": result["seq_len"],
                "D Model": result["d_model"],
                "Dtype": dtype_str,
                "Fwd PT (ms)": f"{result['fwd_pytorch']:.3f}",
                "Fwd TR (ms)": f"{result['fwd_triton']:.3f}",
                "Fwd Speedup": f"{fwd_speedup:.2f}x",
                "Bwd PT (ms)": f"{result['bwd_pytorch']:.3f}",
                "Bwd TR (ms)": f"{result['bwd_triton']:.3f}",
                "Bwd Speedup": f"{bwd_speedup:.2f}x",
                "E2E PT (ms)": f"{result['e2e_pytorch']:.3f}",
                "E2E TR (ms)": f"{result['e2e_triton']:.3f}",
                "E2E Speedup": f"{e2e_speedup:.2f}x",
            }
        )

    df = pd.DataFrame(df_data)

    # Print detailed results table
    print("\n" + "=" * 120)
    print("Detailed Results Table")
    print("=" * 120)
    print(df.to_string(index=False))

    # Generate markdown table
    print("\n" + "=" * 80)
    print("Markdown Table")
    print("=" * 80)
    print(df.to_markdown(index=False))


def quick_test():
    """Quick test with a single configuration"""
    if not torch.cuda.is_available():
        print("CUDA not available. This benchmark requires a GPU.")
        return

    device = "cuda"
    batch_size = 1
    seq_len = 128
    d_model = 64
    dtype = torch.float32
    is_causal = True

    print("Quick Test - Single Configuration")
    print(f"Seq Len: {seq_len}, D Model: {d_model}, Dtype: {dtype}")

    # Generate inputs
    Q, K, V = generate_inputs(batch_size, seq_len, d_model, dtype, device)

    # Test forward passes
    try:
        pytorch_output = attention(Q, K, V, is_causal)
        print("✓ PyTorch forward pass works")
    except Exception as e:
        print(f"✗ PyTorch forward pass failed: {e}")
        return

    try:
        triton_output = FlashAttentionTriton.apply(Q, K, V, is_causal)
        print("✓ Triton forward pass works")
    except Exception as e:
        print(f"✗ Triton forward pass failed: {e}")
        return

    # Quick benchmark
    fwd_pytorch_time = benchmark_forward_pytorch(Q, K, V, is_causal)
    fwd_triton_time = benchmark_forward_triton(Q, K, V, is_causal)

    print(
        f"Forward - PyTorch: {fwd_pytorch_time:.3f}ms, Triton: {fwd_triton_time:.3f}ms"
    )
    print(f"Speedup: {fwd_pytorch_time/fwd_triton_time:.2f}x")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        quick_test()
    else:
        run_benchmark()
