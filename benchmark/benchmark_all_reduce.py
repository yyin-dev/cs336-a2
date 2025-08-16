import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import itertools
import timeit


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def f(rank, world_size, data_size_MB):
    setup(rank, world_size)

    warmup_runs = 2
    benchmark_runs = 10

    n = data_size_MB * 1024 * 1024 // 4
    data = torch.rand((n,))

    # warmup
    for _ in range(warmup_runs):
        dist.all_reduce(data, op=dist.ReduceOp.SUM, async_op=False)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = timeit.default_timer()

    for _ in range(benchmark_runs):
        dist.all_reduce(data, op=dist.ReduceOp.SUM, async_op=False)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    end_time = timeit.default_timer()
    duration = end_time - start_time
    print(f"[{data_size_MB}MB] Rank {rank}: duration {duration}s")


if __name__ == "__main__":
    world_sizes = [2, 4, 6]
    data_size_MBs = [1, 10, 100, 1000]

    for world_size, data_size_MB in itertools.product(world_sizes, data_size_MBs):
        print("=" * 80)
        print(f"world_size: {world_size}, data_size: {data_size_MB}MB")
        mp.spawn(  # pyright: ignore[reportPrivateImportUsage]
            fn=f, args=(world_size, data_size_MB), nprocs=world_size, join=True
        )
