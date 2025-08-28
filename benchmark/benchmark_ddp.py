"""
Benchmark DDP using wrappers.
"""

import argparse
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import random
import timeit
from cs336_systems.model import BasicsTransformerLM
from cs336_systems.nn_utils import cross_entropy
from cs336_systems.optimizer import AdamW
from cs336_systems.ddp_naive import DDPNaive
from cs336_systems.ddp_batch import DDPBatch
from cs336_systems.ddp_overlap_individual_params import DDPOverlapIndividualParams
from cs336_systems.ddp_overlap_bucketed import DDPOverlapBucketed

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


def run(
    rank,
    world_size,
    input_batch,
    output_batch,
    num_steps,
    sequence_len,
    use_xl,
    time_after_backprop,
    mode,
    bucket_size=None,
):
    print(f"[{rank}] starting up")
    device = setup(rank, world_size)

    # Prepare macrobatch
    print(f"[{rank}] preparing data")
    B = input_batch.shape[0]
    start_idx = rank * int(B / world_size)
    end_idx = (rank + 1) * int(B / world_size)
    input_microbatch = input_batch[start_idx:end_idx].to(device)
    output_microbatch = output_batch[start_idx:end_idx].to(device)

    print(f"[{rank}] creating model")
    if torch.cuda.is_available():
        if use_xl:
            # XL
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
            # large
            model = BasicsTransformerLM(
                vocab_size=50527,
                context_length=sequence_len,
                d_model=1280,
                d_ff=5120,
                num_layers=36,
                num_heads=20,
                rope_theta=10000,
            ).to(device)
    else:
        # Just for testing
        model = torch.nn.Sequential(
            torch.nn.Embedding(VOCAB_SIZE, 256),
            torch.nn.Linear(256, VOCAB_SIZE, bias=False),
        )
    print(f"[{rank}] Model created")

    if mode == "naive":
        ddp_model = DDPNaive(model)
    elif mode == "batch":
        ddp_model = DDPBatch(model)
    elif mode == "overlap":
        ddp_model = DDPOverlapIndividualParams(model)
    elif mode == "overlap-bucketed":
        if bucket_size is None:
            raise ValueError("bucket_size must be provided for overlap-bucketed mode")

        ddp_model = DDPOverlapBucketed(model, bucket_size)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    optimizer = AdamW(model.parameters())

    # Warmup
    print(f"[{rank}] Start warming up")
    warmup_steps = 5
    warmup_start = timeit.default_timer()
    for _ in range(warmup_steps):
        optimizer.zero_grad()

        output = ddp_model(input_microbatch)

        loss = cross_entropy(output, output_microbatch)
        loss.backward()

        ddp_model.after_backward()

        optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    warmup_end = timeit.default_timer()
    warmup_duration = warmup_end - warmup_start
    print(f"[{rank}] Warmup duration: {warmup_duration:.3f}s for {warmup_steps} steps")

    # Benchmarking
    training_start = timeit.default_timer()
    total_sync_duration = 0
    for s in range(num_steps):
        optimizer.zero_grad()

        start_time = timeit.default_timer()

        output = ddp_model(input_microbatch)

        loss = cross_entropy(output, output_microbatch)
        loss.backward()

        if time_after_backprop:
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            sync_start = timeit.default_timer()

            ddp_model.after_backward()

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            sync_end = timeit.default_timer()
            sync_duration = sync_end - sync_start
            total_sync_duration += sync_duration
            print(f"[{rank}] synced in {sync_duration:.3f}s")

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        else:
            ddp_model.after_backward()

        optimizer.step()

        end_time = timeit.default_timer()
        duration = end_time - start_time

        if time_after_backprop:
            print(
                f"[{rank}] Training step {s} finished in {duration:.3f}s, gradient sync: {sync_duration:.3f}s, {100*(sync_duration / duration):.3f}%"
            )
        else:
            print(f"[{rank}] Training step {s} finished in {duration:.3f}s")

    training_end = timeit.default_timer()
    duration = training_end - training_start

    if time_after_backprop:
        print(
            f"[{rank}] Training duration: {duration:.3f}s for {num_steps} steps, gradient sync: {total_sync_duration:.3f}s, {100*(total_sync_duration/ duration):.3f}%"
        )
    else:
        print(f"[{rank}] Training duration: {duration:.3f}s for {num_steps} steps")

    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark DDP using wrappers")
    parser.add_argument("--sequence_len", type=int, help="Sequence length")
    parser.add_argument(
        "--use_xl",
        action="store_true",
        default=False,
        help="Use XL model (default: False)",
    )
    parser.add_argument(
        "--time_after_backprop",
        required=True,
        type=lambda x: x.lower() == "true",
        help="Time after backprop (true/false)",
    )
    parser.add_argument(
        "--mode",
        choices=["naive", "batch", "overlap", "overlap-bucketed"],
        default="naive",
        help="DDP mode: naive (DDPNaive), batch (DDPBatch), overlap (DDPOverlapIndividualParams), overlap-bucketed (DDPOverlapBucketed)",
    )
    parser.add_argument(
        "--bucket-size",
        type=float,
        help="Bucket size in MB for DDPOverlapBucketed mode",
    )

    args = parser.parse_args()

    # Validate that bucket-size is provided when mode is overlap-bucketed
    if args.mode == "overlap-bucketed" and args.bucket_size is None:
        parser.error("--bucket-size is required when --mode is 'overlap-bucketed'")

    world_size = 2
    batch_size = 12
    (input_batch, output_batch) = get_data(batch_size, args.sequence_len)

    num_steps = 10
    mp.spawn(  # pyright: ignore[reportPrivateImportUsage]
        fn=run,
        args=(
            world_size,
            input_batch,
            output_batch,
            num_steps,
            args.sequence_len,
            args.use_xl,
            args.time_after_backprop,
            args.mode,
            args.bucket_size,
        ),
        nprocs=world_size,
        join=True,
    )
