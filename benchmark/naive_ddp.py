"""
Toy DDP implementation.
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from copy import deepcopy


import numpy as np
import random


# Seed for deterministic training
seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)


def get_data():
    batch_size = 64
    seq_len = 32
    dim = 16
    data = torch.rand((batch_size, seq_len, dim))
    return data


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    # Needed to run on my MBA sometimes
    # os.environ["GLOO_SOCKET_IFNAME"] = "en0"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    torch.distributed.destroy_process_group()


def naive_ddp(
    rank, world_size, data: torch.Tensor, num_layers, num_steps, check_correctness
):
    setup(rank, world_size)

    B = data.shape[0]
    microbatch_size = int(B / world_size)
    microbatch = data[rank * microbatch_size : (rank + 1) * microbatch_size]

    if rank == 0:
        params = [torch.nn.Parameter(torch.rand((16, 16))) for _ in range(num_layers)]
    else:
        params = [torch.nn.Parameter(torch.empty(0)) for _ in range(num_layers)]

    if check_correctness:
        non_ddp_params = deepcopy(params)
        non_ddp_optimizer = torch.optim.AdamW(non_ddp_params, lr=1e-3)

    # Broadcast initial params from 0 to all else
    dist.broadcast_object_list(params, src=0)
    ddp_optimizer = torch.optim.AdamW(params, lr=1e-3)

    if check_correctness and rank == 0:
        for i in range(num_layers):
            assert torch.allclose(params[i], non_ddp_params[i])

    # DDP SGD on microbatch + Non-DDP SGD on entire batch
    for _ in range(num_steps):
        # DDP
        ddp_x = microbatch
        for param in params:
            ddp_x = ddp_x @ param
            ddp_x = torch.nn.functional.gelu(ddp_x)

        loss = ddp_x.square().mean()
        loss.backward()

        sync_in_batch = True

        if sync_in_batch:

            all_grads_flattened = []
            for param in params:
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
            for param in params:
                assert param.grad is not None
                grad_numel = param.grad.numel()
                grad = torch.reshape(
                    all_grads[start : start + grad_numel], param.grad.shape
                )
                param.grad = grad
                start += grad_numel
        else:
            for param in params:
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

        ddp_optimizer.step()

        # non - ddp
        if check_correctness and rank == 0:
            non_ddp_x = data
            for param in non_ddp_params:
                non_ddp_x = non_ddp_x @ param
                non_ddp_x = torch.nn.functional.gelu(non_ddp_x)

            non_ddp_loss = non_ddp_x.square().mean()
            non_ddp_loss.backward()

            non_ddp_optimizer.step()

            # assert equal
            for i in range(num_layers):
                assert torch.allclose(params[i], non_ddp_params[i])

    cleanup()


if __name__ == "__main__":
    world_size = 2
    data = get_data()
    assert data.shape[0] % world_size == 0

    num_layers = 4
    num_steps = 6
    check_correctness = True
    mp.spawn(  # pyright: ignore[reportPrivateImportUsage]
        fn=naive_ddp,
        args=(world_size, data, num_layers, num_steps, check_correctness),
        nprocs=world_size,
        join=True,
    )
