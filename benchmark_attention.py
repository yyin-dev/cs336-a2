import argparse
import torch.nn as nn
import torch
import timeit
from einops import einsum
import math


def softmax(x, dim=-1):
    rescaled_input = x - torch.max(x, dim=dim, keepdim=True)[0]
    exponentiated_rescaled_input = torch.exp(rescaled_input)
    return exponentiated_rescaled_input / torch.sum(
        exponentiated_rescaled_input, dim=dim, keepdim=True
    )


def scaled_dot_product_attn(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor
):
    """
    Args
        Q: ... m d
        K: ... n d
        V: ... n d
        mask: ... m n

        m and n can be different, to represent one sequence attending to another.
    """
    # naive implementation
    d_k = K.shape[-1]

    attn_scores = einsum(Q, K, "... m d, ... n d -> ... m n") / math.sqrt(d_k)
    attn_scores = torch.where(mask, attn_scores, float("-inf"))

    attn_weights = softmax(attn_scores)
    return einsum(attn_weights, V, "... m n, ... n d -> ... m d")


def get_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"

    return device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--forward-only", action="store_true")

    args = parser.parse_args()

    forward_only = args.forward_only
    print("forward only")

    device = get_device()
    print(f"Device: {device}")

    batch_size = 8
    d_models = [16, 32, 64, 128]
    seq_lens = [256, 1024, 4096, 8192, 16384]

    for d_model in d_models:
        for seq_len in seq_lens:
            shape = (batch_size, seq_len, d_model)

            # Set requires_grad=True so that we can call .backward()
            Q = torch.rand(shape, requires_grad=True, device=device)
            K = torch.rand(shape, requires_grad=True, device=device)
            V = torch.rand(shape, requires_grad=True, device=device)
            mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).bool()

            print(f"== d_model: {d_model}, seq_len: {seq_len} ==")

            # warmup
            for _ in range(10):
                out = scaled_dot_product_attn(Q, K, V, mask)
                loss = out.sum()
                loss.backward()

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # 100 forward passes
            N = 100
            start_time_fwd = timeit.default_timer()
            for _ in range(N):
                out = scaled_dot_product_attn(Q, K, V, mask)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

            end_time_fwd = timeit.default_timer()
            duration_fwd = end_time_fwd - start_time_fwd
            print(f"Forward: {duration_fwd:.4f}s")

            # Meansure memory usage
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated()
                allocated_MB = allocated / (1024**2)
                print(f"CUDA memory usage after forward: {allocated_MB:.4f}MB")

            # 100 backward pass
            out = scaled_dot_product_attn(Q, K, V, mask)
            loss = out.sum()

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            start_time_bwd = timeit.default_timer()
            for _ in range(N):
                # Pass retain_graph=True so we can call .backward() many times.
                # By default, the tensor graph is destroyed after backward.
                loss.backward(retain_graph=True)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

            end_time_bwd = timeit.default_timer()
            duration_bwd = end_time_bwd - start_time_bwd
            print(f"Backward: {duration_bwd:.4f}s")

            # Meansure memory usage
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated()
                allocated_MB = allocated / (1024**2)
                print(f"CUDA memory usage after backward: {allocated_MB:.4f}MB")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
