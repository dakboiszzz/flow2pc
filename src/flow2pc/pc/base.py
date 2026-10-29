import torch
from torch import nn


class ProbabilisticCircuit(nn.Module):
    def path_log_prob(
        self,
        path_values: torch.Tensor,
        times: torch.Tensor,
    ) -> torch.Tensor:
        if path_values.shape[0] != len(times):
            raise ValueError("path_values and times must have the same path length")
        return torch.stack(
            [
                self.log_prob(values, time)
                for values, time in zip(path_values, times, strict=True)
            ]
        )

