"""
Benchmark PyTorch attention vs. Triton FlashAttention.
"""

import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import torch
import triton.testing
import itertools
from einops import einsum, rearrange
import math
import pandas as pd
from cs336_systems.flashattention import FlashAttentionTriton


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


def benchmark_forward_triton(Q, K, V, is_causal=True, q_tile_size=32, k_tile_size=32):
    """Benchmark Triton forward pass"""

    def fn():
        return FlashAttentionTriton.apply(Q, K, V, is_causal, q_tile_size, k_tile_size)

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


def benchmark_end_to_end_triton(
    Q, K, V, is_causal=True, q_tile_size=32, k_tile_size=32
):
    """Benchmark Triton end-to-end forward + backward pass"""

    def fn():
        Q_clone = Q.clone().detach().requires_grad_(True)
        K_clone = K.clone().detach().requires_grad_(True)
        V_clone = V.clone().detach().requires_grad_(True)
        O = FlashAttentionTriton.apply(
            Q_clone, K_clone, V_clone, is_causal, q_tile_size, k_tile_size
        )
        loss = O.sum()
        loss.backward()

    return triton.testing.do_bench(fn)


# Define tile size configurations to test
def get_tile_configs(seq_len, d_model):
    """Get comprehensive tile configurations for given sequence length and head dimension"""
    configs = []

    # Base tile sizes to consider
    tile_sizes = [16, 32, 64, 128]

    if seq_len <= 256:
        # Small sequences: test smaller tiles for better occupancy
        configs = [(16, 16), (16, 32), (32, 16), (32, 32), (32, 64), (64, 32)]
    elif seq_len <= 1024:
        # Medium sequences: balanced approach
        configs = [
            (16, 16),
            (32, 32),
            (64, 32),
            (32, 64),
            (64, 64),
            (128, 32),
            (32, 128),
        ]
    elif seq_len <= 4096:
        # Large sequences: favor larger tiles to reduce kernel launches
        configs = [
            (32, 32),
            (64, 32),
            (32, 64),
            (64, 64),
            (128, 32),
            (32, 128),
            (128, 64),
            (64, 128),
        ]
    else:
        # Very large sequences: use largest tiles
        configs = [
            (64, 32),
            (32, 64),
            (64, 64),
            (128, 32),
            (32, 128),
            (128, 64),
            (64, 128),
            (128, 128),
        ]

    # Filter configs based on d_model to avoid excessive register usage
    if d_model > 128:
        # For large d_model, prefer smaller tiles to fit in registers
        configs = [(q, k) for q, k in configs if q <= 64 and k <= 64]
    elif d_model > 64:
        # For medium d_model, limit tile sizes moderately
        configs = [(q, k) for q, k in configs if q <= 128 and k <= 128]

    # Remove duplicates while preserving order
    seen = set()
    filtered_configs = []
    for config in configs:
        if config not in seen:
            seen.add(config)
            filtered_configs.append(config)

    return filtered_configs


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
    dtypes = [torch.float32, torch.bfloat16]

    print("FlashAttention Benchmark Results")
    print("=" * 80)
    print(f"Batch Size: {batch_size}, Causal Masking: {is_causal}")
    print("=" * 80)

    results = []

    for seq_len, d_model, dtype in itertools.product(seq_lengths, d_models, dtypes):
        try:
            # Generate inputs
            Q, K, V = generate_inputs(batch_size, seq_len, d_model, dtype, device)

            # Get tile configurations to test
            tile_configs = get_tile_configs(seq_len, d_model)

            # Warm up GPU
            for _ in range(3):
                _ = attention(Q, K, V, is_causal)
                _ = FlashAttentionTriton.apply(Q, K, V, is_causal, 32, 32)

            torch.cuda.synchronize()

            # Benchmark PyTorch (only once per config)
            fwd_pytorch_time = benchmark_forward_pytorch(Q, K, V, is_causal)
            e2e_pytorch_time = benchmark_end_to_end_pytorch(Q, K, V, is_causal)
            bwd_pytorch_time = e2e_pytorch_time - fwd_pytorch_time

            # Test each tile configuration
            best_fwd_time = float("inf")
            best_config = None
            config_results = []

            for q_tile, k_tile in tile_configs:
                # Benchmark forward pass
                fwd_triton_time = benchmark_forward_triton(
                    Q, K, V, is_causal, q_tile, k_tile
                )

                # Benchmark end-to-end (forward + backward)
                e2e_triton_time = benchmark_end_to_end_triton(
                    Q, K, V, is_causal, q_tile, k_tile
                )

                # Calculate backward time by subtraction
                bwd_triton_time = e2e_triton_time - fwd_triton_time

                # Track best configuration
                if fwd_triton_time < best_fwd_time:
                    best_fwd_time = fwd_triton_time
                    best_config = (q_tile, k_tile)

                config_results.append(
                    {
                        "seq_len": seq_len,
                        "d_model": d_model,
                        "dtype": dtype,
                        "q_tile_size": q_tile,
                        "k_tile_size": k_tile,
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
                    f"Dtype: {str(dtype).split('.')[-1]:8s}, Tiles: ({q_tile:2d},{k_tile:2d}) - "
                    f"Fwd: PT {fwd_pytorch_time:.3f}ms / TR {fwd_triton_time:.3f}ms, "
                    f"Bwd: PT {bwd_pytorch_time:.3f}ms / TR {bwd_triton_time:.3f}ms, "
                    f"E2E: PT {e2e_pytorch_time:.3f}ms / TR {e2e_triton_time:.3f}ms"
                )

            print(f"  → Best tile config for this setup: {best_config}")

            # Add results with best tile marking
            for config_result in config_results:
                is_best = (
                    config_result["q_tile_size"],
                    config_result["k_tile_size"],
                ) == best_config
                config_result["is_best"] = is_best
                results.append(config_result)

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
                "Q Tile": result["q_tile_size"],
                "K Tile": result["k_tile_size"],
                "Best": "✓" if result["is_best"] else "",
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

    # Create best-only table (filter to only best configurations)
    best_df = df[df["Best"] == "✓"].copy()

    # Print best configurations table
    print("\n" + "=" * 80)
    print("Best Configurations Only")
    print("=" * 80)
    print(best_df.to_string(index=False))

    # Generate markdown table for best configurations
    print("\n" + "=" * 80)
    print("Best Configurations Markdown")
    print("=" * 80)
    print(best_df.to_markdown(index=False))

    # Print detailed results table
    print("\n" + "=" * 120)
    print("Detailed Results Table")
    print("=" * 120)
    print(df.to_string(index=False))

    # Generate markdown table for all results
    print("\n" + "=" * 80)
    print("All Results Markdown Table")
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
    is_causal = True

    # Test both dtypes
    dtypes_to_test = [torch.float32, torch.bfloat16]

    for dtype in dtypes_to_test:
        print(f"\nQuick Test - Single Configuration")
        print(f"Seq Len: {seq_len}, D Model: {d_model}, Dtype: {dtype}")

        # Generate inputs
        Q, K, V = generate_inputs(batch_size, seq_len, d_model, dtype, device)

        # Test forward passes
        try:
            pytorch_output = attention(Q, K, V, is_causal)
            print("✓ PyTorch forward pass works")
        except Exception as e:
            print(f"✗ PyTorch forward pass failed: {e}")
            continue

        try:
            triton_output = FlashAttentionTriton.apply(Q, K, V, is_causal)
            print("✓ Triton forward pass works")

            # Check outputs are close
            max_diff = torch.max(torch.abs(pytorch_output - triton_output))
            print(f"✓ Max difference between outputs: {max_diff:.6f}")

        except Exception as e:
            print(f"✗ Triton forward pass failed: {e}")
            continue

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
