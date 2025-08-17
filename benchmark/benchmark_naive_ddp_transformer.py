"""
DDP on Transformer XL, seq_len = 256.
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import random
import timeit
from cs336_systems.model import BasicsTransformerLM
from cs336_systems.nn_utils import cross_entropy
from cs336_systems.optimizer import AdamW

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
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def get_device(rank):
    if torch.cuda.is_available():
        return f"cuda:{rank}"
    else:
        return "cpu"


def ddp_transformer_xl(rank, world_size, input_batch, output_batch, num_steps):
    setup(rank, world_size)

    # Prepare macrobatch

    B = input_batch.shape[0]
    start_idx = rank * int(B / world_size)
    end_idx = (rank + 1) * int(B / world_size)
    input_microbatch = input_batch[start_idx:end_idx].to(get_device(rank))
    output_microbatch = output_batch[start_idx:end_idx].to(get_device(rank))

    if torch.cuda.is_available():
        model = BasicsTransformerLM(
            vocab_size=50527,
            context_length=256,
            d_model=1600,
            d_ff=6400,
            num_layers=48,
            num_heads=25,
            rope_theta=10000,
        ).to(get_device(rank))
    else:
        # Just for testing
        model = torch.nn.Sequential(
            torch.nn.Embedding(VOCAB_SIZE, 256),
            torch.nn.Linear(256, VOCAB_SIZE, bias=False),
        )

    # Sync initial weights
    for param in model.parameters():
        dist.broadcast(param.data, src=0, async_op=False)

    optimizer = AdamW(model.parameters())

    # Warmup
    warmup_steps = 5
    warmup_start = timeit.default_timer()
    for _ in range(warmup_steps):
        optimizer.zero_grad()

        output = model(input_microbatch)
        loss = cross_entropy(output, output_microbatch)
        loss.backward()

        for param in model.parameters():
            if dist.get_backend() == "gloo":
                dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.SUM, async_op=False)
                param.grad /= world_size
            else:
                dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG, async_op=False)

        optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    warmup_end = timeit.default_timer()
    warmup_duration = warmup_end - warmup_start
    print(f"[{rank}] Warmup duration: {warmup_duration:.2f}s for {warmup_steps} steps")

    # Benchmarking
    training_start = timeit.default_timer()
    total_sync_duration = 0
    for s in range(num_steps):
        optimizer.zero_grad()

        start_time = timeit.default_timer()
        output = model(input_microbatch)
        loss = cross_entropy(output, output_microbatch)
        loss.backward()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # sync gradients
        sync_start = timeit.default_timer()
        sync_in_batch = True

        if sync_in_batch:
            # all-reduce all gradients in one batch
            all_grads_flattened = []
            for param in model.parameters():
                assert param.grad is not None
                flattened_grad = torch.flatten(param.grad)
                all_grads_flattened.append(flattened_grad)

            all_grads = torch.concat(all_grads_flattened)
            if dist.get_backend() == "gloo":
                # Gloo doesn't support AVG
                dist.all_reduce(tensor=all_grads, op=dist.ReduceOp.SUM, async_op=False)
                all_grads /= world_size
            else:
                dist.all_reduce(tensor=all_grads, op=dist.ReduceOp.AVG, async_op=False)

            start = 0
            for param in model.parameters():
                assert param.grad is not None
                grad_numel = param.grad.numel()
                grad = torch.reshape(
                    all_grads[start : start + grad_numel], param.grad.shape
                )
                param.grad = grad
                start += grad_numel
        else:
            # all-reduce each gradient separately
            for param in model.parameters():
                if dist.get_backend() == "gloo":
                    # Gloo doesn't support AVG
                    dist.all_reduce(
                        tensor=param.grad, op=dist.ReduceOp.SUM, async_op=False
                    )
                    param.grad /= world_size
                else:
                    dist.all_reduce(
                        tensor=param.grad, op=dist.ReduceOp.AVG, async_op=False
                    )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        sync_end = timeit.default_timer()
        sync_duration = sync_end - sync_start
        total_sync_duration += sync_duration
        print(f"[{rank}] Gradient synced in {sync_duration:.2f}s")

        optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = timeit.default_timer()
        duration = end_time - start_time
        print(
            f"[{rank}] Training step {s} finished in {duration:.2f}s, gradient sync: {100*(sync_duration / duration):.2f}%"
        )

    training_end = timeit.default_timer()
    duration = training_end - training_start
    print(
        f"[{rank}] Training duration: {duration:.2f}s for {num_steps} steps, gradient sync: {100*(total_sync_duration/ duration):.2f}%"
    )

    cleanup()


if __name__ == "__main__":
    world_size = 2
    batch_size = 8
    seq_len = 256
    (input_batch, output_batch) = get_data(batch_size, seq_len)

    num_steps = 10
    mp.spawn(  # pyright: ignore[reportPrivateImportUsage]
        fn=ddp_transformer_xl,
        args=(world_size, input_batch, output_batch, num_steps),
        nprocs=world_size,
        join=True,
    )
