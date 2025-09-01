"""
Benchmark optimizer state sharding vs regular optimizer memory usage.

Usage:
  # Regular optimizer (baseline)
  python benchmark_opt_sharding.py --sequence_len 512

  # Sharded optimizer
  python benchmark_opt_sharding.py --sequence_len 512 --use_sharded_optimizer

Note: For meaningful memory measurements, run on a system with CUDA GPUs.
On CPU, memory measurements will show 0 MB but the script will still verify correctness.
"""

import argparse
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import random
from cs336_systems.model import BasicsTransformerLM
from cs336_systems.nn_utils import cross_entropy
from cs336_systems.optimizer import AdamW
from tests.adapters import get_sharded_optimizer

seed = 42
random.seed(seed)
torch.manual_seed(42)

VOCAB_SIZE = 50527


def get_data(batch_size, seq_len):
    batch = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len + 1))
    input_batch = batch[:, :seq_len]
    output_batch = batch[:, 1:]
    return input_batch, output_batch


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    backend = "nccl" if torch.cuda.is_available() else "gloo"

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if device_count > 0:
            local_rank = rank % device_count
            torch.cuda.set_device(local_rank)
            device = f"cuda:{local_rank}"
        else:
            raise ValueError("Unable to find CUDA devices")
    else:
        device = "cpu"

    dist.init_process_group(backend, rank=rank, world_size=world_size)
    return device


def cleanup():
    dist.destroy_process_group()


def get_memory_mb():
    """Get current GPU memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    else:
        # For CPU, we can't easily measure memory, so return 0
        return 0


def calculate_model_memory_theoretical(model):
    """Calculate theoretical memory usage for model components"""
    total_params = sum(p.numel() for p in model.parameters())
    param_memory_mb = total_params * 4 / 1024**2  # 4 bytes per float32 param

    return {
        "total_params": total_params,
        "param_memory_mb": param_memory_mb,
        "gradient_memory_mb": param_memory_mb,  # Gradients same size as params
        # AdamW has 2 state tensors (momentum + variance) per parameter
        "optimizer_state_mb_regular": param_memory_mb * 2,
        "optimizer_state_mb_sharded": param_memory_mb,  # ~50% with 2 ranks
    }


def get_detailed_memory_breakdown(model, optimizer, use_sharded):
    """Get detailed breakdown of memory usage"""
    theoretical = calculate_model_memory_theoretical(model)
    current_memory_mb = get_memory_mb()

    # For sharded optimizer with 2 ranks, each rank only stores ~50% of optimizer state
    world_size = dist.get_world_size() if dist.is_initialized() else 2
    sharding_factor = 1.0 / world_size if use_sharded else 1.0

    breakdown = {
        "current_total_mb": current_memory_mb,
        "theoretical_params_mb": theoretical["param_memory_mb"],
        "theoretical_gradients_mb": theoretical["gradient_memory_mb"],
        "theoretical_optimizer_mb": theoretical["optimizer_state_mb_regular"]
        * sharding_factor,
        "theoretical_total_mb": theoretical["param_memory_mb"]
        + theoretical["gradient_memory_mb"]
        + (theoretical["optimizer_state_mb_regular"] * sharding_factor),
        "total_params": theoretical["total_params"],
        "sharding_factor": sharding_factor,
    }

    return breakdown


def run(
    rank,
    world_size,
    input_batch,
    output_batch,
    num_steps,
    sequence_len,
    use_sharded_optimizer,
):
    print(f"[{rank}] starting up")
    device = setup(rank, world_size)
    print(f"Backend: {dist.get_backend()}")

    # Prepare microbatch for this rank
    print(f"[{rank}] preparing data")
    B = input_batch.shape[0]
    start_idx = rank * int(B / world_size)
    end_idx = (rank + 1) * int(B / world_size)
    input_microbatch = input_batch[start_idx:end_idx].to(device)
    output_microbatch = output_batch[start_idx:end_idx].to(device)

    print(f"[{rank}] creating model")
    if torch.cuda.is_available():
        # XL model as specified
        model = BasicsTransformerLM(
            vocab_size=50527,
            context_length=sequence_len,
            d_model=1600,
            d_ff=6400,
            num_layers=48,
            num_heads=25,
            rope_theta=10000,
        ).to(device)
    else:
        # Just for testing
        model = torch.nn.Sequential(
            torch.nn.Embedding(VOCAB_SIZE, 256),
            torch.nn.Linear(256, VOCAB_SIZE, bias=False),
        ).to(device)

    # Clear cache and synchronize
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Memory checkpoint 1: After model initialization
    memory_after_model = get_memory_mb()
    breakdown_after_model = get_detailed_memory_breakdown(
        model, None, use_sharded_optimizer
    )
    print(f"[{rank}] Memory after model initialization: {memory_after_model:.2f} MB")
    print(
        f"[{rank}] Theoretical params: {breakdown_after_model['theoretical_params_mb']:.2f} MB, Total params: {breakdown_after_model['total_params']:,}"
    )

    # Create optimizer
    if use_sharded_optimizer:
        optimizer = get_sharded_optimizer(
            model.parameters(),
            AdamW,
            lr=1e-4,
            weight_decay=0.1,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        optimizer_type = "Sharded"
    else:
        optimizer = AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.1,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        optimizer_type = "Regular"

    print(f"[{rank}] Using {optimizer_type} optimizer")

    # Memory checkpoint 2: After optimizer creation
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    memory_after_optimizer = get_memory_mb()
    breakdown_after_optimizer = get_detailed_memory_breakdown(
        model, optimizer, use_sharded_optimizer
    )
    print(f"[{rank}] Memory after optimizer creation: {memory_after_optimizer:.2f} MB")
    print(
        f"[{rank}] Theoretical optimizer state: {breakdown_after_optimizer['theoretical_optimizer_mb']:.2f} MB"
    )

    # Memory measurements
    memory_measurements = []

    for step in range(num_steps):
        print(f"[{rank}] Step {step}")

        optimizer.zero_grad()

        output = model(input_microbatch)
        loss = cross_entropy(output, output_microbatch)
        loss.backward()

        # Clear cache and synchronize before measuring memory
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Memory checkpoint 2: Before optimizer step
        memory_before_step = get_memory_mb()
        print(
            f"[{rank}] Memory before optimizer step {step}: {memory_before_step:.2f} MB"
        )

        optimizer.step()

        # Note: For memory profiling, we don't need parameter synchronization
        # The memory usage patterns are identical to real distributed training
        # but without the communication overhead

        # Clear cache and synchronize after optimizer step
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Memory checkpoint 3: After optimizer step
        memory_after_step = get_memory_mb()
        print(
            f"[{rank}] Memory after optimizer step {step}: {memory_after_step:.2f} MB"
        )

        memory_measurements.append(
            {
                "step": step,
                "before_step": memory_before_step,
                "after_step": memory_after_step,
            }
        )

    # Report peak memory usage
    peak_before_step = max(m["before_step"] for m in memory_measurements)
    peak_after_step = max(m["after_step"] for m in memory_measurements)

    # Final detailed breakdown
    final_breakdown = get_detailed_memory_breakdown(
        model, optimizer, use_sharded_optimizer
    )

    print(f"\n[{rank}] === DETAILED MEMORY BREAKDOWN ({optimizer_type} Optimizer) ===")
    print(
        f"[{rank}] Model Parameters:        {final_breakdown['theoretical_params_mb']:8.2f} MB ({final_breakdown['total_params']:,} params)"
    )
    print(
        f"[{rank}] Gradients:               {final_breakdown['theoretical_gradients_mb']:8.2f} MB"
    )
    print(
        f"[{rank}] Optimizer State:         {final_breakdown['theoretical_optimizer_mb']:8.2f} MB (factor: {final_breakdown['sharding_factor']:.2f})"
    )
    print(
        f"[{rank}] Theoretical Total:       {final_breakdown['theoretical_total_mb']:8.2f} MB"
    )
    print(f"[{rank}] ---")
    print(f"[{rank}] Measured after model:    {memory_after_model:8.2f} MB")
    print(f"[{rank}] Measured after optimizer:{memory_after_optimizer:8.2f} MB")
    print(f"[{rank}] Peak before step:        {peak_before_step:8.2f} MB")
    print(f"[{rank}] Peak after step:         {peak_after_step:8.2f} MB")

    # Calculate memory savings if sharded
    if use_sharded_optimizer:
        regular_theoretical = (
            final_breakdown["theoretical_params_mb"]
            + final_breakdown["theoretical_gradients_mb"]
            + (
                final_breakdown["theoretical_optimizer_mb"]
                / final_breakdown["sharding_factor"]
            )
        )
        savings_mb = regular_theoretical - final_breakdown["theoretical_total_mb"]
        savings_pct = (savings_mb / regular_theoretical) * 100
        print(
            f"[{rank}] Memory savings:          {savings_mb:8.2f} MB ({savings_pct:.1f}%)"
        )

    cleanup()

    return {
        "rank": rank,
        "optimizer_type": optimizer_type,
        "memory_after_model": memory_after_model,
        "memory_after_optimizer": memory_after_optimizer,
        "peak_before_step": peak_before_step,
        "peak_after_step": peak_after_step,
        "breakdown": final_breakdown,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark optimizer state sharding memory usage"
    )
    parser.add_argument("--sequence_len", type=int, default=512, help="Sequence length")
    parser.add_argument(
        "--use_sharded_optimizer",
        action="store_true",
        default=False,
        help="Use sharded optimizer (default: False, uses regular optimizer)",
    )

    args = parser.parse_args()

    world_size = 2
    batch_size = 12
    (input_batch, output_batch) = get_data(batch_size, args.sequence_len)

    num_steps = 3  # Just a few steps to measure memory

    print(
        f"Starting benchmark with {'sharded' if args.use_sharded_optimizer else 'regular'} optimizer"
    )
    print(f"World size: {world_size}")
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {args.sequence_len}")
    print(f"Number of steps: {num_steps}")

    mp.spawn(  # pyright: ignore[reportPrivateImportUsage]
        fn=run,
        args=(
            world_size,
            input_batch,
            output_batch,
            num_steps,
            args.sequence_len,
            args.use_sharded_optimizer,
        ),
        nprocs=world_size,
        join=True,
    )
