import torch
import torch.nn as nn
import torch.distributed as dist


# Seed for deterministic training
seed = 42
torch.manual_seed(seed)


# Sync individual gradients. Overlap computation with communication.
#
# Don't spawn processes in this class. Assume torch.multiprocessing.spawn
# has been called already. Each process will create a wrapper. This class
# should detect the DDP setup like rank and world size.
#
# See the testcases for how the class is used.
class DDPOverlapIndividualParams(nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self.handles = []

        def on_grad(param: torch.Tensor) -> None:
            if param.grad is None:
                return

            if dist.get_backend() == "gloo":
                handle = dist.all_reduce(
                    param.grad, op=dist.ReduceOp.SUM, async_op=True
                )
                param.grad /= dist.get_world_size()
            else:
                handle = dist.all_reduce(
                    param.grad, op=dist.ReduceOp.AVG, async_op=True
                )

            self.handles.append(handle)

        for param in module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(on_grad)

        # Broadcast initial weights from 0
        for param in module.parameters():
            dist.broadcast(param.data, src=0, async_op=False)

        dist.barrier()

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()

        self.handles.clear()

    # For my benchmarking purpose.
    def after_backward(self):
        self.finish_gradient_synchronization()
