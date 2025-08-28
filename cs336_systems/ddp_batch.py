import torch
import torch.nn as nn
import torch.distributed as dist

# Seed for deterministic training
seed = 42
torch.manual_seed(seed)


# Sync all params in one go. Don't overlap compute with communication.
class DDPBatch(nn.Module):
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
        all_grads = [
            param.grad for param in self.module.parameters() if param.grad is not None
        ]

        flattened = torch._utils._flatten_dense_tensors(all_grads)

        if dist.get_backend() == "gloo":
            # Gloo doesn't support AVG
            dist.all_reduce(tensor=flattened, op=dist.ReduceOp.SUM, async_op=False)
            flattened /= dist.get_world_size()
        else:
            dist.all_reduce(tensor=flattened, op=dist.ReduceOp.AVG, async_op=False)

        all_grads = torch._utils._unflatten_dense_tensors(flattened, all_grads)

        idx = 0
        for param in self.module.parameters():
            if param.grad is not None:
                param.grad = all_grads[idx]
                idx += 1
