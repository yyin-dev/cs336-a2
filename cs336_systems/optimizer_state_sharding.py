import torch
import torch.distributed as dist
from typing import Type, Any


class OptimizerStateSharding(torch.optim.Optimizer):
    def __init__(
        self, params, optimizer_cls: Type[torch.optim.Optimizer], **kwargs: Any
    ):
        self.optimizer_cls = optimizer_cls
        self.optim: torch.optim.Optimizer | None = None

        # super()'s constructor calls add_param_group
        self.params = list(params)
        super().__init__(self.params, defaults=kwargs)

        # determine the shard
        # TODO: take param size into account
        W = dist.get_world_size()

        shard_size = (len(self.params) + W - 1) // W
        rank = dist.get_rank()
        shard_start = rank * shard_size
        shard_end = (rank + 1) * shard_size
        shard_params = self.params[shard_start:shard_end]

    def step(self, closure=None, **kwargs):  # type: ignore
        assert self.optim is not None
        self.optim.step(closure, **kwargs)

        W = dist.get_world_size()
        for i in range(W):
            shard_size = (len(self.params) + W - 1) // W
            shard_start = i * shard_size
            shard_end = (i + 1) * shard_size
            dist.broadcast_object_list(self.params[shard_start:shard_end], src=i)

    def add_param_group(self, param_group: dict[str, Any]):
        print(f"[{dist.get_rank()}] add param group")
        self.param_groups.append(param_group)

        if not self.optim:
            self.optim = self.optimizer_cls([param_group], **self.defaults)
        else:
            self.optim.add_param_group(param_group)
