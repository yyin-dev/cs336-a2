import torch
import torch.nn as nn
import torch.distributed as dist

# Seed for deterministic training
seed = 42
torch.manual_seed(seed)


# Sync individual params. Don't overlap compute with communication.
class DDPNaive(nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module

        # Broadcast initial weights from 0
        for param in module.parameters():
            dist.broadcast(param.data, src=0, async_op=False)

        dist.barrier()

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def after_backward(self):
        for param in self.module.parameters():
            if param.grad is not None:
                if dist.get_backend() == "gloo":
                    # Gloo doesn't support AVG
                    dist.all_reduce(
                        tensor=param.grad, op=dist.ReduceOp.SUM, async_op=False
                    )
                    param.grad /= dist.get_world_size()
                else:
                    dist.all_reduce(
                        tensor=param.grad, op=dist.ReduceOp.AVG, async_op=False
                    )
