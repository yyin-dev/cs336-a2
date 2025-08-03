from cs336_basics.model import BasicsTransformerLM as Transformer
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_basics.annotated_model import BasicsTransformerLM as AnnotatedTransformer
import argparse
from contextlib import nullcontext

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
    run_backward: bool,
    run_optimizer: bool,
    profile: bool,
    mixed_precision: bool,
):
    TransformerClass = AnnotatedTransformer if profile else Transformer

    model = TransformerClass(
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

    targets = torch.randint(
        low=0,
        high=10000,
        size=(batch_size, context_length),
        device=get_device(),
    )

    optimizer = AdamW(model.parameters())
    dtype = torch.bfloat16

    optional_cast_context_manager = (
        torch.autocast(device_type=get_device(), dtype=dtype)
        if mixed_precision
        else nullcontext()
    )

    if not run_backward:

        def run():
            with optional_cast_context_manager:
                with nvtx.range("forward"):
                    output = model.forward(input)

    else:
        if not run_optimizer:

            def run():

                with optional_cast_context_manager:
                    with nvtx.range("forward"):
                        output = model.forward(input)

                    with nvtx.range("loss"):
                        loss = cross_entropy(output, targets)

                    with nvtx.range("backward"):
                        loss.backward()

        else:

            def run():
                with optional_cast_context_manager:
                    optimizer.zero_grad()

                    with nvtx.range("forward"):
                        output = model.forward(input)

                    with nvtx.range("loss"):
                        loss = cross_entropy(output, targets)

                    with nvtx.range("backward"):
                        loss.backward()

                    with nvtx.range("optimizer"):
                        optimizer.step()

    with nvtx.range("warmup"):
        for _ in range(num_warmups):
            run()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    times: list[float] = []

    if profile:
        # When nsys profiling, we run fewer trials and capture the profile
        with nvtx.range("nsys_profiling"):
            for _ in range(min(3, num_trials)):  # Limit trials for profiling
                model.zero_grad()

                start_time = timeit.default_timer()
                run()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = timeit.default_timer()
                times.append((end_time - start_time))
    else:
        # Normal timing runs
        for _ in range(num_trials):
            model.zero_grad()

            start_time = timeit.default_timer()
            run()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = timeit.default_timer()
            times.append((end_time - start_time))

    mean_time = statistics.mean(times)
    std = statistics.stdev(times) if len(times) > 1 else 0.0
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
    parser.add_argument("--forward_backward", action="store_true")
    parser.add_argument("--forward_backward_and_optimizer", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--mixed_precision", action="store_true")

    args = parser.parse_args()
    print(args)

    if args.cpu:
        print("Warning: Benchmarking on CPU")
    else:
        assert torch.cuda.is_available()

    if args.profile:
        print(f"Note: nsys profiling enabled")

    if args.forward_only:
        print(f"forward only")
        run_backward = False
        run_optimizer = False
    elif args.forward_backward:
        print(f"forward backward")
        run_backward = True
        run_optimizer = False
    elif args.forward_backward_and_optimizer:
        print(f"forward backward and optimizer")
        run_backward = True
        run_optimizer = True
    else:
        raise ValueError("No option selected")

    if args.mixed_precision:
        print("Mixed precision")
    else:
        print("Full precision")

    mean, std = benchmark(
        args.batch_size,
        args.num_layers,
        args.context_length,
        args.d_model,
        args.num_heads,
        args.d_ff,
        args.num_warmups,
        args.num_trials,
        run_backward,
        run_optimizer,
        args.profile,
        args.mixed_precision,
    )
    print(f"Mean: {mean:.4f}")
    print(f"Std: {std:.4f}")
    print(f"Std/Mean={std/mean:.2f}")


if __name__ == "__main__":
    main()
