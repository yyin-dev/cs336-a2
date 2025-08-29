import torch
import torch.nn as nn
import torch.distributed as dist


# Seed for deterministic training
seed = 42
torch.manual_seed(seed)


class Bucket:
    def __init__(self, start, end, num_params):
        self.start = start  # index in model.parameters(), inclusive
        self.end = end  # index in model.parameters(), exclusive
        self.num_params = num_params

        self.grad_cnt = 0
        self.handle: dist.Work | None = None
        self.flattened_grad: torch.Tensor | None = None

    def is_empty(self):
        return self.start == self.end

    def reset(self):
        self.grad_cnt = 0
        self.handle = None
        self.flattened_grad = None

    def __repr__(self):
        return str(self.__dict__)


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
        self.params = list(self.module.parameters())
        self.buckets: list[Bucket] = []

        # Find buckets
        # self.buckets is in reverse order w.r.t module.parmeters()
        current_bucket = Bucket(len(self.params), len(self.params), 0)
        current_bucket_size = 0
        for i in range(len(self.params) - 1, -1, -1):
            param = self.params[i]

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
                current_bucket.num_params += 1
            else:
                prev_bucket = current_bucket
                self.buckets.append(prev_bucket)

                current_bucket = Bucket(i, prev_bucket.start, 1)
                current_bucket_size = grad_size

        if current_bucket_size > 0:
            self.buckets.append(current_bucket)

        # print_params(self.module.parameters())
        # print(f"[{dist.get_rank()}] buckets: {self.buckets}")

        # Set up hooks on backprop
        for bucket in self.buckets:

            # It's important to pass in [bucket] as argument here. O/w we
            # run into the classical Python closure trap: all lambdas share
            # the same bucket (the last value after the loop ends)
            def on_grad(_param: torch.Tensor, bucket: Bucket = bucket) -> None:
                bucket.grad_cnt += 1

                if bucket.grad_cnt < bucket.num_params:
                    return

                # bucket.grad_cnt == bucket.num_params
                if bucket.start + 1 == bucket.end:
                    bucket.flattened_grad = self.params[bucket.start].grad
                else:
                    bucket_params = self.params[bucket.start : bucket.end]
                    grads = [p.grad for p in bucket_params if p.requires_grad]
                    bucket.flattened_grad = torch._utils._flatten_dense_tensors(grads)

                bucket.handle = dist.all_reduce(
                    tensor=bucket.flattened_grad,
                    op=dist.ReduceOp.SUM,
                    async_op=True,
                )

            for i in range(bucket.start, bucket.end):
                param = self.params[i]

                if not param.requires_grad:
                    continue

                param.register_post_accumulate_grad_hook(on_grad)

        # Broadcast initial weights from 0
        for param in module.parameters():
            dist.broadcast(param.data, src=0, async_op=False)

        dist.barrier()

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for bucket in self.buckets:
            # assert bucket.handle is not None
            # assert bucket.flattened_grad is not None

            bucket.handle.wait()
            bucket.flattened_grad /= dist.get_world_size()

            if bucket.start + 1 == bucket.end:
                self.params[bucket.start].grad = bucket.flattened_grad
            else:
                bucket_params = self.params[bucket.start : bucket.end]
                grads = [param.grad for param in bucket_params if param.requires_grad]
                grads = torch._utils._unflatten_dense_tensors(
                    bucket.flattened_grad, grads
                )

                idx = 0
                for param in bucket_params:
                    if param.requires_grad:
                        # assert param.grad is not None
                        param.grad = grads[idx]
                        idx += 1

            bucket.reset()

    # For my benchmarking purpose.
    def after_backward(self):
        self.finish_gradient_synchronization()
