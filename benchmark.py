from cs336_basics.model import BasicsTransformerLM
import argparse

import torch
import numpy as np
import random
import timeit
import statistics
import torch.cuda.nvtx as nvtx

# Seed for determinism
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


def get_device():
    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


def benchmark(
    batch_size: int,
    num_layers: int,
    context_length: int,
    d_model: int,
    num_heads: int,
    d_ff: int,
    num_warmups: int,
    num_trials: int,
    forward_only: bool,
):
    model = BasicsTransformerLM(
        vocab_size=10000,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=10000,
    )
    model.to(get_device())

    input = torch.randint(
        low=0,
        high=10000,
        size=(batch_size, context_length),
        device=get_device(),
    )

    if forward_only:

        def run():
            output = model.forward(input).mean()

    else:

        def run():
            output = model.forward(input).mean()
            output.backward()

    with nvtx.range("warmup"):
        for _ in range(num_warmups):
            run()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(num_trials):
        start_time = timeit.default_timer()
        run()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = timeit.default_timer()
        times.append((end_time - start_time))

    mean_time = statistics.mean(times)
    std = statistics.stdev(times)
    return mean_time, std


def main():
    parser = argparse.ArgumentParser()

    # hyperparameters
    parser.add_argument("--context_length", type=int, required=True)
    parser.add_argument("--d_model", type=int, required=True)
    parser.add_argument("--num_layers", type=int, required=True)
    parser.add_argument("--num_heads", type=int, required=True)
    parser.add_argument("--d_ff", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--num_warmups", type=int, default=5)
    parser.add_argument("--num_trials", type=int, default=10)
    parser.add_argument("--forward_only", action="store_true")

    args = parser.parse_args()
    print(args)

    if args.cpu:
        print("Warning: Benchmarking on CPU")
    else:
        assert torch.cuda.is_available()

    mean, std = benchmark(
        args.batch_size,
        args.num_layers,
        args.context_length,
        args.d_model,
        args.num_heads,
        args.d_ff,
        args.num_warmups,
        args.num_trials,
        args.forward_only,
    )
    print(f"Mean: {mean:.4f}")
    print(f"Std: {std:.4f}")
    print(f"Std/Mean={std/mean:.2f}")


if __name__ == "__main__":
    main()
