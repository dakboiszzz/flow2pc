import math

import torch
from torch import nn

from repc.pc import ProbabilisticCircuit

LOG_2PI = math.log(2.0 * math.pi)


def build_region_graph(height: int, width: int):
    splits_by_region = {}

    def visit(row_start: int, row_end: int, col_start: int, col_end: int):
        region = (row_start, row_end, col_start, col_end)
        if region in splits_by_region:
            return region
        splits_by_region[region] = []
        if row_end - row_start > 1:
            middle = (row_start + row_end) // 2
            splits_by_region[region].append(
                (
                    visit(row_start, middle, col_start, col_end),
                    visit(middle, row_end, col_start, col_end),
                )
            )
        if col_end - col_start > 1:
            middle = (col_start + col_end) // 2
            splits_by_region[region].append(
                (
                    visit(row_start, row_end, col_start, middle),
                    visit(row_start, row_end, middle, col_end),
                )
            )
        return region

    root = visit(0, height, 0, width)
    regions = list(splits_by_region)
    leaves = [region for region in regions if not splits_by_region[region]]
    leaves.sort(key=lambda region: region[0] * width + region[2])
    internal = [region for region in regions if splits_by_region[region]]
    internal.sort(
        key=lambda region: (
            (region[1] - region[0]) * (region[3] - region[2]),
            region,
        )
    )
    ordered = leaves + internal
    position = {region: index for index, region in enumerate(ordered)}
    splits = [
        [(position[left], position[right]) for left, right in splits_by_region[region]]
        for region in ordered
    ]
    return ordered, splits, position[root]


class RegionGraphPC(ProbabilisticCircuit):
    """Hierarchical PC over horizontal and vertical image partitions."""

    def __init__(
        self,
        height: int = 28,
        width: int = 28,
        num_channels: int = 8,
        time_order: int = 4,
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.dim = height * width
        self.num_channels = num_channels
        self.time_order = time_order
        regions, splits, root = build_region_graph(height, width)
        self.num_leaves = self.dim

        regions_by_shape = {}
        for index in range(self.num_leaves, len(regions)):
            row_start, row_end, col_start, col_end = regions[index]
            shape = (row_end - row_start, col_end - col_start)
            regions_by_shape.setdefault(shape, []).append(index)

        remap = torch.full((len(regions),), -1, dtype=torch.long)
        remap[: self.num_leaves] = torch.arange(self.num_leaves)
        offset = self.num_leaves
        self.groups = []
        ordered_shapes = sorted(
            regions_by_shape,
            key=lambda item: (item[0] * item[1], item),
        )
        for shape in ordered_shapes:
            indices = regions_by_shape[shape]
            num_splits = len(splits[indices[0]])
            child_left = remap[
                torch.tensor([[pair[0] for pair in splits[index]] for index in indices])
            ]
            child_right = remap[
                torch.tensor([[pair[1] for pair in splits[index]] for index in indices])
            ]
            if (child_left < 0).any() or (child_right < 0).any():
                raise RuntimeError("region graph is not topologically ordered")
            buffer_positions = list(range(offset, offset + len(indices)))
            remap[torch.tensor(indices)] = torch.tensor(buffer_positions)
            offset += len(indices)
            group_index = len(self.groups)
            self.register_buffer(
                f"child_left_{group_index}", child_left, persistent=False
            )
            self.register_buffer(
                f"child_right_{group_index}", child_right, persistent=False
            )
            self.groups.append(
                {
                    "size": len(indices),
                    "num_splits": num_splits,
                    "buffer_positions": buffer_positions,
                }
            )

        self.root_position = int(remap[root])
        channels = num_channels
        self.sum_logits = nn.ParameterList(
            [
                nn.Parameter(
                    0.01
                    * torch.randn(
                        group["size"],
                        channels,
                        group["num_splits"] * channels * channels,
                    )
                )
                for group in self.groups
            ]
        )
        self.root_logits = nn.Parameter(torch.zeros(channels))
        coefficient_shape = (self.dim, channels, time_order)
        self.mean_coefficients = nn.Parameter(0.01 * torch.randn(coefficient_shape))
        self.log_scale_coefficients = nn.Parameter(
            0.01 * torch.randn(coefficient_shape)
        )

    def _time_basis(self, time: torch.Tensor):
        powers = torch.arange(
            1,
            self.time_order + 1,
            device=time.device,
            dtype=time.dtype,
        )
        return time**powers, powers * time ** (powers - 1)

    def leaf_parameters(self, time: torch.Tensor):
        basis, derivative = self._time_basis(time)
        mean = torch.einsum("dsp,p->ds", self.mean_coefficients, basis)
        mean_dt = torch.einsum("dsp,p->ds", self.mean_coefficients, derivative)
        log_scale = torch.einsum("dsp,p->ds", self.log_scale_coefficients, basis)
        log_scale_dt = torch.einsum(
            "dsp,p->ds", self.log_scale_coefficients, derivative
        )
        return mean, mean_dt, log_scale, log_scale_dt

    def _leaf_log_prob(self, values: torch.Tensor, time: torch.Tensor):
        mean, _, log_scale, _ = self.leaf_parameters(time)
        normalized = (values[:, :, None] - mean[None]) * torch.exp(-log_scale)[None]
        return -0.5 * (normalized.square() + LOG_2PI) - log_scale[None]

    def _circuit_log_prob(self, leaf_log_prob: torch.Tensor):
        batch_size = leaf_log_prob.shape[0]
        buffer = leaf_log_prob
        for group_index, group in enumerate(self.groups):
            child_left = getattr(self, f"child_left_{group_index}")
            child_right = getattr(self, f"child_right_{group_index}")
            left = buffer[:, child_left]
            right = buffer[:, child_right]
            products = (left[..., :, None] + right[..., None, :]).reshape(
                batch_size, group["size"], -1
            )
            log_weights = torch.log_softmax(self.sum_logits[group_index], dim=-1)
            output = torch.logsumexp(
                log_weights[None] + products[:, :, None, :], dim=-1
            )
            buffer = torch.cat((buffer, output), dim=1)
        root = buffer[:, self.root_position]
        return torch.logsumexp(
            torch.log_softmax(self.root_logits, dim=0)[None] + root, dim=1
        )

    def log_prob(self, values: torch.Tensor, time: torch.Tensor):
        return self._circuit_log_prob(self._leaf_log_prob(values, time))

    def velocity(self, values: torch.Tensor, time: torch.Tensor):
        mean, mean_dt, _, log_scale_dt = self.leaf_parameters(time)
        with torch.enable_grad():
            leaf_log_prob = self._leaf_log_prob(values, time)
            if not leaf_log_prob.requires_grad:
                leaf_log_prob.requires_grad_(True)
            root_log_prob = self._circuit_log_prob(leaf_log_prob)
            flows = torch.autograd.grad(
                root_log_prob.sum(),
                leaf_log_prob,
                create_graph=self.training,
            )[0]
        leaf_velocity = mean_dt[None] + log_scale_dt[None] * (
            values[:, :, None] - mean[None]
        )
        return (flows * leaf_velocity).sum(dim=-1)

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        time: torch.Tensor,
        generator: torch.Generator | None = None,
    ):
        return self.sample_path(num_samples, time.reshape(1), generator)[0]

    @torch.no_grad()
    def sample_path(
        self,
        num_samples: int,
        times: torch.Tensor,
        generator: torch.Generator | None = None,
    ):
        weights = [torch.softmax(logits, dim=-1) for logits in self.sum_logits]
        root_weights = torch.softmax(self.root_logits, dim=0)
        position_to_group = {
            position: (group_index, row)
            for group_index, group in enumerate(self.groups)
            for row, position in enumerate(group["buffer_positions"])
        }
        device = self.mean_coefficients.device
        dtype = self.mean_coefficients.dtype
        leaf_channels = torch.empty(
            num_samples,
            self.dim,
            device=device,
            dtype=torch.long,
        )
        noise = torch.randn(
            num_samples,
            self.dim,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        channels = self.num_channels
        for sample_index in range(num_samples):
            channel = torch.multinomial(
                root_weights, 1, generator=generator
            ).item()
            stack = [(self.root_position, channel)]
            while stack:
                position, channel = stack.pop()
                if position < self.num_leaves:
                    leaf_channels[sample_index, position] = channel
                    continue
                group_index, row = position_to_group[position]
                selected = torch.multinomial(
                    weights[group_index][row, channel], 1, generator=generator
                ).item()
                split_index, remainder = divmod(selected, channels * channels)
                left_channel, right_channel = divmod(remainder, channels)
                child_left = getattr(self, f"child_left_{group_index}")
                child_right = getattr(self, f"child_right_{group_index}")
                stack.append((int(child_left[row, split_index]), left_channel))
                stack.append((int(child_right[row, split_index]), right_channel))

        variables = torch.arange(self.dim, device=device)[None]
        path = []
        for time in times:
            mean, _, log_scale, _ = self.leaf_parameters(time)
            selected_mean = mean[variables, leaf_channels]
            selected_scale = torch.exp(log_scale[variables, leaf_channels])
            path.append(selected_mean + selected_scale * noise)
        return torch.stack(path)
