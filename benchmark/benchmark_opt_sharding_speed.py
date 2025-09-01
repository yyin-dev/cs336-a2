"""
Benchmark optimizer state sharding vs regular optimizer training speed.

Usage:
  # Regular optimizer (baseline)
  python benchmark_opt_sharding_speed.py --sequence_len 512
  
  # Sharded optimizer 
  python benchmark_opt_sharding_speed.py --sequence_len 512 --use_sharded_optimizer

Note: For meaningful timing measurements, run on a system with CUDA GPUs.
"""

import argparse
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import random
import time
import statistics
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


def run_timing_benchmark(
    rank,
    world_size,
    input_batch,
    output_batch,
    warmup_steps,
    timing_steps,
    sequence_len,
    use_sharded_optimizer,
):
    if rank == 0:
        print(f"Starting up distributed training...")
    device = setup(rank, world_size)
    if rank == 0:
        print(f"Backend: {dist.get_backend()}")

    # Prepare microbatch for this rank
    if rank == 0:
        print("Preparing data...")
    B = input_batch.shape[0]
    start_idx = rank * int(B / world_size)
    end_idx = (rank + 1) * int(B / world_size)
    input_microbatch = input_batch[start_idx:end_idx].to(device)
    output_microbatch = output_batch[start_idx:end_idx].to(device)

    if rank == 0:
        print("Creating model...")
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

    if rank == 0:
        print(f"Using {optimizer_type} optimizer")

    # Warmup phase
    if rank == 0:
        print(f"Starting warmup ({warmup_steps} steps)")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    warmup_start = time.time()
    for step in range(warmup_steps):
        optimizer.zero_grad()
        output = model(input_microbatch)
        loss = cross_entropy(output, output_microbatch)
        loss.backward()
        optimizer.step()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    warmup_end = time.time()
    warmup_duration = warmup_end - warmup_start
    if rank == 0:
        print(f"Warmup completed: {warmup_duration:.3f}s for {warmup_steps} steps")

    # Timing phase
    if rank == 0:
        print(f"Starting timing benchmark ({timing_steps} steps)")
    step_times = []
    
    for step in range(timing_steps):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        step_start = time.time()
        
        # Forward pass timing
        forward_start = time.time()
        optimizer.zero_grad()
        output = model(input_microbatch)
        loss = cross_entropy(output, output_microbatch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        forward_end = time.time()
        forward_time = forward_end - forward_start
        
        # Backward pass timing
        backward_start = time.time()
        loss.backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        backward_end = time.time()
        backward_time = backward_end - backward_start
        
        # Optimizer step timing
        optimizer_start = time.time()
        optimizer.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        optimizer_end = time.time()
        optimizer_time = optimizer_end - optimizer_start
        
        step_end = time.time()
        total_step_time = step_end - step_start
        
        step_times.append({
            'total': total_step_time,
            'forward': forward_time,
            'backward': backward_time,
            'optimizer': optimizer_time,
        })
        
        if rank == 0:  # Only print from rank 0
            print(f"Step {step}: {total_step_time:.4f}s (fwd: {forward_time:.4f}s, bwd: {backward_time:.4f}s, opt: {optimizer_time:.4f}s)")

    # Calculate statistics
    total_times = [t['total'] for t in step_times]
    forward_times = [t['forward'] for t in step_times]
    backward_times = [t['backward'] for t in step_times]
    optimizer_times = [t['optimizer'] for t in step_times]
    
    avg_total = statistics.mean(total_times)
    std_total = statistics.stdev(total_times) if len(total_times) > 1 else 0
    avg_forward = statistics.mean(forward_times)
    avg_backward = statistics.mean(backward_times)
    avg_optimizer = statistics.mean(optimizer_times)
    
    # Only print results from rank 0 to avoid parsing conflicts
    if rank == 0:
        print(f"\n=== TIMING RESULTS ({optimizer_type} Optimizer) ===")
        print(f"Steps measured: {timing_steps}")
        print(f"Average total time:     {avg_total:.4f}s ± {std_total:.4f}s")
        print(f"Average forward time:   {avg_forward:.4f}s ({100*avg_forward/avg_total:.1f}%)")
        print(f"Average backward time:  {avg_backward:.4f}s ({100*avg_backward/avg_total:.1f}%)")
        print(f"Average optimizer time: {avg_optimizer:.4f}s ({100*avg_optimizer/avg_total:.1f}%)")
        print(f"Throughput: {1.0/avg_total:.2f} steps/sec")

    cleanup()

    return {
        'rank': rank,
        'optimizer_type': optimizer_type,
        'avg_total_time': avg_total,
        'std_total_time': std_total,
        'avg_forward_time': avg_forward,
        'avg_backward_time': avg_backward,
        'avg_optimizer_time': avg_optimizer,
        'throughput': 1.0 / avg_total,
        'all_times': step_times,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark optimizer state sharding training speed")
    parser.add_argument("--sequence_len", type=int, default=512, help="Sequence length")
    parser.add_argument(
        "--use_sharded_optimizer",
        action="store_true",
        default=False,
        help="Use sharded optimizer (default: False, uses regular optimizer)",
    )
    parser.add_argument("--warmup_steps", type=int, default=5, help="Number of warmup steps")
    parser.add_argument("--timing_steps", type=int, default=20, help="Number of steps to measure timing")

    args = parser.parse_args()

    world_size = 2
    batch_size = 12
    (input_batch, output_batch) = get_data(batch_size, args.sequence_len)

    print(f"Starting timing benchmark with {'sharded' if args.use_sharded_optimizer else 'regular'} optimizer")
    print(f"World size: {world_size}")
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {args.sequence_len}")
    print(f"Warmup steps: {args.warmup_steps}")
    print(f"Timing steps: {args.timing_steps}")

    mp.spawn(  # pyright: ignore[reportPrivateImportUsage]
        fn=run_timing_benchmark,
        args=(
            world_size,
            input_batch,
            output_batch,
            args.warmup_steps,
            args.timing_steps,
            args.sequence_len,
            args.use_sharded_optimizer,
        ),
        nprocs=world_size,
        join=True,
    )