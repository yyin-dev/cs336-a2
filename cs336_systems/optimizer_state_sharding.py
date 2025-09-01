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
        super().__init__(params, defaults=kwargs)

    def step(self, closure=None, **kwargs):  # type: ignore
        assert self.optim is not None
        self.optim.step(closure, **kwargs)

        W = dist.get_world_size()

        cnt = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    src = cnt % W
                    dist.broadcast(p.data, src, async_op=False)

                cnt += 1

    def add_param_group(self, param_group: dict[str, Any]):
        # Assumes that param_group["params"] is a list of params
        assert param_group["params"] is not None
        assert isinstance(param_group["params"], list)

        # Find the shard within [param_group]
        num_existing_params = sum([len(g["params"]) for g in self.param_groups])
        param_group_to_add = {"params": []}
        for idx, param in enumerate(param_group["params"]):
            global_idx = idx + num_existing_params
            if global_idx % dist.get_world_size() == dist.get_rank():
                param_group_to_add["params"].append(param)

        if len(param_group_to_add["params"]) == 0:
            return

        if not self.optim:
            self.optim = self.optimizer_cls([param_group_to_add], **self.defaults)
        else:
            self.optim.add_param_group(param_group_to_add)

        # Must add param_group, instead of param_groups_to_add, to self.params_group s.t.
        # 1. optimizer.zero_grad() works
        # 2. we hold all parameters for broadcast in step().
        self.param_groups.append(param_group)
