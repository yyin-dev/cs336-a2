import torch
import torch.nn as nn
import torch.distributed as dist
from dataclasses import dataclass


# Seed for deterministic training
seed = 42
torch.manual_seed(seed)


@dataclass
class Bucket:
    start: int  # index in model.parameters(), inclusive
    end: int  # index in model.parameters(), exclusive
    handle: dist.Work | None
    flattened_grad: torch.Tensor | None

    def is_empty(self):
        return self.start == self.end


def print_params(params):
    params_info = [
        (i, p.requires_grad, p.data.numel() * p.data.element_size())
        for i, p in enumerate(params)
    ]
    print(params_info)


# Sync gradients in buckets. Overlap computation with communication.
class DDPOverlapBucketed(nn.Module):
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float):
        super().__init__()
        self.module = module
        self.buckets: list[Bucket] = []

        # Find buckets
        # self.buckets is in reverse order w.r.t module.parmeters()
        params = list(module.parameters())
        current_bucket = Bucket(len(params), len(params), None, None)
        current_bucket_size = 0
        for i in range(len(params) - 1, -1, -1):
            param = params[i]

            if not param.requires_grad:
                continue

            # Grad size should be the same as data size
            grad_size = param.data.numel() * param.data.element_size()

            if (
                current_bucket_size + grad_size <= bucket_size_mb * (2**20)
                or current_bucket.is_empty()
            ):
                current_bucket_size += grad_size
                current_bucket.start = i
            else:
                prev_bucket = current_bucket
                self.buckets.append(prev_bucket)

                current_bucket = Bucket(i, prev_bucket.start, None, None)
                current_bucket_size = grad_size

        if current_bucket_size > 0:
            self.buckets.append(current_bucket)

        print_params(self.module.parameters())
        print(f"[{dist.get_rank()}] buckets: {self.buckets}")

        # Set up hooks on backprop
        for bucket in self.buckets:

            def on_grad(param: torch.Tensor, bucket: Bucket = bucket) -> None:
                # It's important to pass in [bucket] as argument here. O/w we
                # run into the classical Python closure trap: all lambdas share
                # the same bucket (the last value after the loop ends)
                params_in_bucket = params[bucket.start : bucket.end]
                all_grads = [
                    param.grad for param in params_in_bucket if param.grad is not None
                ]

                bucket.flattened_grad = torch._utils._flatten_dense_tensors(all_grads)
                bucket.handle = dist.all_reduce(
                    tensor=bucket.flattened_grad, op=dist.ReduceOp.SUM, async_op=True
                )

            start_param = params[bucket.start]
            start_param.register_post_accumulate_grad_hook(on_grad)

        # Broadcast initial weights from 0
        for param in module.parameters():
            dist.broadcast(param.data, src=0, async_op=False)

        dist.barrier()

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for bucket in self.buckets:
            assert bucket.handle is not None
            assert bucket.flattened_grad is not None

            bucket.handle.wait()
            bucket.flattened_grad /= dist.get_world_size()

            params_in_bucket = list(self.module.parameters())[bucket.start : bucket.end]
            all_grads = [
                param.grad for param in params_in_bucket if param.grad is not None
            ]

            all_grads = torch._utils._unflatten_dense_tensors(
                bucket.flattened_grad, all_grads
            )

            idx = 0
            for param in params_in_bucket:
                if param.grad is not None:
                    param.grad = all_grads[idx]
                    idx += 1

            bucket.handle = None
            bucket.flattened_grad = None

    # For my benchmarking purpose.
    def after_backward(self):
        self.finish_gradient_synchronization()
