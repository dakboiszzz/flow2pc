import math
from collections import deque

import torch
from torch import nn

from flow2pc.pc.gaussian_pc import ProbabilisticCircuit, discretized_gaussian_log_mass

LOG_2PI = math.log(2.0 * math.pi)


def _tree_schedule(parents: list[int]) -> tuple[int, list[list[int]], list[list[int]]]:
    """Validate a rooted tree and group nodes by bottom-up computation level."""
    dim = len(parents)
    roots = [node for node, parent in enumerate(parents) if parent == -1]
    if len(roots) != 1:
        raise ValueError("an HCLT topology must have exactly one root")
    root = roots[0]
    children = [[] for _ in range(dim)]
    for node, parent in enumerate(parents):
        if parent == -1:
            continue
        if parent < 0 or parent >= dim or parent == node:
            raise ValueError(f"invalid parent {parent} for node {node}")
        children[parent].append(node)

    remaining_children = [len(node_children) for node_children in children]
    current = [node for node, count in enumerate(remaining_children) if count == 0]
    levels = []
    visited = 0
    while current:
        levels.append(current)
        visited += len(current)
        next_level = []
        for node in current:
            parent = parents[node]
            if parent != -1:
                remaining_children[parent] -= 1
                if remaining_children[parent] == 0:
                    next_level.append(parent)
        current = next_level
    if visited != dim or levels[-1] != [root]:
        raise ValueError("HCLT parents must form one connected acyclic tree")
    return root, children, levels


class TimeDependentHCLTPC(ProbabilisticCircuit):
    """Hidden Chow--Liu tree with time-conditioned Gaussian-mixture emissions.

    The hidden-tree compilation follows the HCLT construction used by PyJuice:
    each image variable has one latent state, child-to-parent transitions are sum
    layers, and emissions and child messages meet in Hadamard product layers.
    ``sum_layer='monarch'`` replaces every dense transition by the three
    block-diagonal layers (a composition of two Monarch matrices) from the
    official MonarchCircuits implementation.
    """

    def __init__(
        self,
        parents: list[int],
        num_latents: int = 256,
        time_order: int = 4,
        sum_layer: str = "dense",
        time_conditioning: str = "polynomial",
        time_embedding_dim: int = 16,
        time_feature_seed: int = 0,
        leaf_distribution: str = "gaussian",
        leaf_components: int = 1,
        num_categories: int = 256,
        fourier_zero_center: bool = True,
    ):
        super().__init__()
        if num_latents < 2:
            raise ValueError("num_latents must be at least two")
        if time_order < 1:
            raise ValueError("time_order must be positive")
        if sum_layer not in ("dense", "monarch"):
            raise ValueError("sum_layer must be 'dense' or 'monarch'")
        if time_conditioning not in ("polynomial", "fourier"):
            raise ValueError("time_conditioning must be 'polynomial' or 'fourier'")
        if time_embedding_dim < 2 or time_embedding_dim % 2:
            raise ValueError("time_embedding_dim must be a positive even integer")
        if leaf_distribution not in ("gaussian", "gaussian-mixture", "categorical"):
            raise ValueError(
                "leaf_distribution must be 'gaussian', 'gaussian-mixture', "
                "or 'categorical'"
            )
        if leaf_distribution == "gaussian" and leaf_components != 1:
            raise ValueError("Gaussian leaves require exactly one component")
        if leaf_distribution == "gaussian-mixture" and leaf_components < 2:
            raise ValueError("Gaussian-mixture leaves require at least two components")
        if leaf_distribution == "categorical" and leaf_components != 1:
            raise ValueError("Categorical leaves do not use mixture components")
        if num_categories < 2:
            raise ValueError("num_categories must be at least two")

        self.dim = len(parents)
        self.num_latents = num_latents
        self.time_order = time_order
        self.sum_layer = sum_layer
        self.time_conditioning = time_conditioning
        self.time_embedding_dim = time_embedding_dim
        self.time_feature_seed = time_feature_seed
        self.leaf_distribution = leaf_distribution
        self.leaf_components = leaf_components
        self.num_categories = num_categories
        self.fourier_zero_center = fourier_zero_center
        root, children, levels = _tree_schedule(parents)
        self.root = root
        self.tree_children = children
        self.levels = levels
        self.register_buffer("parents", torch.tensor(parents, dtype=torch.long))

        non_root_nodes = [node for node in range(self.dim) if node != root]
        edge_by_node = torch.full((self.dim,), -1, dtype=torch.long)
        edge_by_node[non_root_nodes] = torch.arange(self.dim - 1)
        self.register_buffer("edge_by_node", edge_by_node, persistent=False)

        channels = num_latents
        if sum_layer == "dense":
            self.transition_logits = nn.Parameter(
                0.01 * torch.randn(self.dim - 1, channels, channels)
            )
            self.monarch_logits = None
        else:
            block_size = math.isqrt(channels)
            if block_size * block_size != channels:
                raise ValueError("Monarch HCLT requires a square number of latents")
            self.block_size = block_size
            self.num_blocks = channels // block_size
            self.monarch_logits = nn.Parameter(
                0.01
                * torch.randn(
                    self.dim - 1,
                    3,
                    self.num_blocks,
                    block_size,
                    block_size,
                )
            )
            self.transition_logits = None
            permutation = (
                torch.arange(channels)
                .reshape(self.num_blocks, block_size)
                .T.reshape(-1)
            )
            self.register_buffer("monarch_permutation", permutation, persistent=False)

        self.root_logits = nn.Parameter(torch.zeros(channels))
        self.categorical_logits = None
        if leaf_distribution == "categorical":
            # Match the reference DLTM initialization: independent Dirichlet(1)
            # transition/root rows and random categorical emission logits.
            with torch.no_grad():
                root_probabilities = torch.empty(channels).exponential_()
                root_probabilities /= root_probabilities.sum()
                self.root_logits.copy_(root_probabilities.log())
                if self.transition_logits is not None:
                    transition_probabilities = torch.empty_like(
                        self.transition_logits
                    ).exponential_()
                    transition_probabilities /= transition_probabilities.sum(
                        dim=-1,
                        keepdim=True,
                    )
                    self.transition_logits.copy_(transition_probabilities.log())
            self.categorical_logits = nn.Parameter(
                torch.randn(self.dim, channels, num_categories)
            )
            self.leaf_mixture_logits = None
            self.mean_coefficients = None
            self.log_scale_coefficients = None
            return

        if time_conditioning == "fourier":
            # Match the fixed Gaussian Fourier feature layer used by PICs. The
            # cosine branch is centered so every feature is zero at t=0 and
            # the PC path therefore starts at exactly N(0, I).
            self.register_buffer(
                "time_frequencies",
                torch.randn(
                    time_embedding_dim // 2,
                    generator=torch.Generator().manual_seed(time_feature_seed),
                ),
            )
            num_time_features = time_embedding_dim
        else:
            self.register_buffer(
                "time_frequencies",
                torch.empty(0),
                persistent=False,
            )
            num_time_features = time_order
        coefficient_shape = (self.dim, channels, num_time_features)
        if leaf_distribution == "gaussian-mixture":
            coefficient_shape = (
                self.dim,
                channels,
                leaf_components,
                num_time_features,
            )
            self.leaf_mixture_logits = nn.Parameter(
                torch.zeros(self.dim, channels, leaf_components)
            )
        else:
            self.leaf_mixture_logits = None
        self.mean_coefficients = nn.Parameter(0.01 * torch.randn(coefficient_shape))
        self.log_scale_coefficients = nn.Parameter(
            0.01 * torch.randn(coefficient_shape)
        )

    def _time_basis(self, time: torch.Tensor):
        if self.time_conditioning == "fourier":
            scalar_time = time[:, None] if time.ndim > 0 else time[None]
            angular_frequencies = 2.0 * math.pi * self.time_frequencies
            angles = scalar_time * angular_frequencies
            cos_term = (
                torch.cos(angles) - 1.0
                if self.fourier_zero_center
                else torch.cos(angles)
            )
            basis = torch.stack(
                (cos_term, torch.sin(angles)),
                dim=-1,
            ).flatten(-2)
            derivative = torch.stack(
                (
                    -angular_frequencies * torch.sin(angles),
                    angular_frequencies * torch.cos(angles),
                ),
                dim=-1,
            ).flatten(-2)
            return basis, derivative

        powers = torch.arange(
            1,
            self.time_order + 1,
            device=time.device,
            dtype=time.dtype,
        )
        if time.ndim > 0:
            time = time[:, None]
        return time**powers, powers * time ** (powers - 1)

    def leaf_parameters(self, time: torch.Tensor):
        if self.leaf_distribution == "categorical":
            raise RuntimeError("categorical HCLT leaves only define the t=1 endpoint")
        basis, derivative = self._time_basis(time)
        if self.leaf_distribution == "gaussian-mixture" and time.ndim == 0:
            mean = torch.einsum("dhkp,p->dhk", self.mean_coefficients, basis)
            mean_dt = torch.einsum("dhkp,p->dhk", self.mean_coefficients, derivative)
            log_scale = torch.einsum("dhkp,p->dhk", self.log_scale_coefficients, basis)
            log_scale_dt = torch.einsum(
                "dhkp,p->dhk", self.log_scale_coefficients, derivative
            )
        elif self.leaf_distribution == "gaussian-mixture":
            mean = torch.einsum("dhkp,bp->bdhk", self.mean_coefficients, basis)
            mean_dt = torch.einsum("dhkp,bp->bdhk", self.mean_coefficients, derivative)
            log_scale = torch.einsum(
                "dhkp,bp->bdhk", self.log_scale_coefficients, basis
            )
            log_scale_dt = torch.einsum(
                "dhkp,bp->bdhk", self.log_scale_coefficients, derivative
            )
        elif time.ndim == 0:
            mean = torch.einsum("dhp,p->dh", self.mean_coefficients, basis)
            mean_dt = torch.einsum("dhp,p->dh", self.mean_coefficients, derivative)
            log_scale = torch.einsum("dhp,p->dh", self.log_scale_coefficients, basis)
            log_scale_dt = torch.einsum(
                "dhp,p->dh", self.log_scale_coefficients, derivative
            )
        else:
            mean = torch.einsum("dhp,bp->bdh", self.mean_coefficients, basis)
            mean_dt = torch.einsum("dhp,bp->bdh", self.mean_coefficients, derivative)
            log_scale = torch.einsum("dhp,bp->bdh", self.log_scale_coefficients, basis)
            log_scale_dt = torch.einsum(
                "dhp,bp->bdh", self.log_scale_coefficients, derivative
            )
        return mean, mean_dt, log_scale, log_scale_dt

    def _leaf_component_parameters(self, time: torch.Tensor):
        parameters = self.leaf_parameters(time)
        if time.ndim == 0:
            parameters = tuple(parameter[None] for parameter in parameters)
        if self.leaf_distribution == "gaussian":
            parameters = tuple(parameter[..., None] for parameter in parameters)
        return parameters

    @staticmethod
    def _component_log_prob(
        values: torch.Tensor,
        mean: torch.Tensor,
        log_scale: torch.Tensor,
    ) -> torch.Tensor:
        normalized = (values[:, :, None, None] - mean) * torch.exp(-log_scale)
        return -0.5 * (normalized.square() + LOG_2PI) - log_scale

    def _mix_leaf_components(self, component_log_prob: torch.Tensor) -> torch.Tensor:
        if self.leaf_distribution == "gaussian":
            return component_log_prob.squeeze(-1)
        log_weights = torch.log_softmax(self.leaf_mixture_logits, dim=-1)
        return torch.logsumexp(log_weights[None] + component_log_prob, dim=-1)

    def _leaf_log_prob(self, values: torch.Tensor, time: torch.Tensor):
        if self.leaf_distribution == "categorical":
            if time.ndim != 0 or not torch.isclose(
                time,
                time.new_tensor(1.0),
            ):
                raise ValueError("categorical HCLT likelihood is defined only at t=1")
            categories = self._category_indices(
                values, self.categorical_logits.shape[-1]
            )
            log_probabilities = torch.log_softmax(self.categorical_logits, dim=-1)
            lookup = log_probabilities.permute(0, 2, 1)
            variables = torch.arange(self.dim, device=values.device)[None]
            return lookup[variables, categories]
        mean, _, log_scale, _ = self._leaf_component_parameters(time)
        component_log_prob = self._component_log_prob(values, mean, log_scale)
        return self._mix_leaf_components(component_log_prob)

    @staticmethod
    def _category_indices(values: torch.Tensor, num_categories: int) -> torch.Tensor:
        if values.is_floating_point():
            categories = torch.round(
                (values.clamp(-1.0, 1.0) + 1.0) * (num_categories - 1) / 2.0
            )
        else:
            categories = values
        return categories.long().clamp(0, num_categories - 1)

    @staticmethod
    def _dense_sum(scores: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """Apply many independent normalized dense sum layers in log space."""
        maximum = scores.max(dim=-1, keepdim=True).values
        probabilities = torch.exp(scores - maximum)
        weights = torch.softmax(logits, dim=-1)
        output = torch.bmm(probabilities, weights.transpose(1, 2))
        tiny = torch.finfo(output.dtype).tiny
        return torch.log(output.clamp_min(tiny)) + maximum

    def _monarch_sum(
        self,
        scores: torch.Tensor,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the official three-layer Monarch HCLT transition."""
        for layer in range(3):
            if layer > 0:
                scores = scores.index_select(-1, self.monarch_permutation)
            blocked = scores.reshape(
                scores.shape[0], scores.shape[1], self.num_blocks, self.block_size
            )
            maximum = blocked.max(dim=-1, keepdim=True).values
            probabilities = torch.exp(blocked - maximum)
            weights = torch.softmax(logits[:, layer], dim=-1)
            output = torch.einsum("enoi,ebni->ebno", weights, probabilities)
            tiny = torch.finfo(output.dtype).tiny
            scores = (torch.log(output.clamp_min(tiny)) + maximum).flatten(2)
        return scores

    def _circuit_log_prob(self, leaf_log_prob: torch.Tensor) -> torch.Tensor:
        incoming: list[list[torch.Tensor]] = [[] for _ in range(self.dim)]
        for nodes in self.levels:
            node_scores = []
            for node in nodes:
                score = leaf_log_prob[:, node]
                if incoming[node]:
                    score = score + torch.stack(incoming[node]).sum(dim=0)
                node_scores.append(score)
            scores = torch.stack(node_scores)
            if nodes == [self.root]:
                log_weights = torch.log_softmax(self.root_logits, dim=0)
                return torch.logsumexp(log_weights[None] + scores[0], dim=-1)

            node_tensor = torch.tensor(nodes, device=self.parents.device)
            edge_indices = self.edge_by_node[node_tensor]
            if self.sum_layer == "dense":
                messages = self._dense_sum(scores, self.transition_logits[edge_indices])
            else:
                messages = self._monarch_sum(scores, self.monarch_logits[edge_indices])
            for index, node in enumerate(nodes):
                incoming[int(self.parents[node])].append(messages[index])
        raise RuntimeError("invalid HCLT evaluation schedule")

    def log_prob(self, values: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        return self._circuit_log_prob(self._leaf_log_prob(values, time))

    def endpoint_em_statistics(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute exact soft-EM sufficient statistics at the data endpoint.

        Derivatives with respect to leaf log-likelihoods are posterior latent
        probabilities. For a normalized dense transition row, adding its
        logit gradient to the parent posterior count times the current row
        probability recovers the exact parent-child posterior counts.
        """
        if self.sum_layer != "dense":
            raise ValueError("endpoint EM currently requires dense HCLT layers")
        if self.leaf_distribution not in ("gaussian", "categorical"):
            raise ValueError("endpoint EM requires Gaussian or categorical HCLT leaves")

        endpoint = torch.tensor(1.0, device=values.device, dtype=values.dtype)
        with torch.enable_grad():
            leaf_log_prob = self._leaf_log_prob(values, endpoint)
            log_prob = self._circuit_log_prob(leaf_log_prob)
            leaf_flows, transition_gradient = torch.autograd.grad(
                log_prob.sum(),
                (leaf_log_prob, self.transition_logits),
            )

        with torch.no_grad():
            node_by_edge = torch.nonzero(self.parents >= 0).flatten()
            parent_by_edge = self.parents.index_select(0, node_by_edge)
            parent_counts = leaf_flows.index_select(1, parent_by_edge).sum(dim=0)
            transition_probabilities = torch.softmax(
                self.transition_logits,
                dim=-1,
            )
            transition_counts = transition_gradient + (
                parent_counts[:, :, None] * transition_probabilities
            )
            transition_counts.clamp_(min=0.0)
            statistics = {
                "log_likelihood": log_prob.detach().sum(),
                "num_examples": torch.tensor(
                    values.shape[0],
                    device=values.device,
                    dtype=leaf_flows.dtype,
                ),
                "root_counts": leaf_flows[:, self.root].sum(dim=0),
                "transition_counts": transition_counts,
                "leaf_counts": leaf_flows.sum(dim=0),
            }
            if self.leaf_distribution == "categorical":
                categories = self._category_indices(
                    values,
                    self.categorical_logits.shape[-1],
                )
                category_counts = torch.zeros_like(self.categorical_logits)
                category_counts.scatter_add_(
                    2,
                    categories.T[:, None].expand(-1, self.num_latents, -1),
                    leaf_flows.permute(1, 2, 0),
                )
                statistics["leaf_category_counts"] = category_counts
            else:
                statistics["leaf_sums"] = torch.einsum("bdh,bd->dh", leaf_flows, values)
                statistics["leaf_square_sums"] = torch.einsum(
                    "bdh,bd->dh",
                    leaf_flows,
                    values.square(),
                )
            return statistics

    @torch.no_grad()
    def endpoint_em_update(
        self,
        statistics: dict[str, torch.Tensor],
        pseudocount: float = 0.1,
        min_std: float = 0.01,
        step_size: float = 1.0,
    ) -> None:
        """Apply a soft-EM update at ``t=1``.

        ``step_size=1`` is an exact full-batch M-step. Smaller values implement
        the stochastic mini-batch EM update used by the published HCLT code.
        """
        if self.sum_layer != "dense":
            raise ValueError("endpoint EM currently requires dense HCLT layers")
        if self.leaf_distribution not in ("gaussian", "categorical"):
            raise ValueError("endpoint EM requires Gaussian or categorical HCLT leaves")
        if pseudocount < 0.0:
            raise ValueError("EM pseudocount must be non-negative")
        if min_std <= 0.0:
            raise ValueError("EM minimum standard deviation must be positive")
        if not 0.0 < step_size <= 1.0:
            raise ValueError("EM step size must be in (0, 1]")

        root_counts = statistics["root_counts"] + pseudocount
        root_probabilities = root_counts / root_counts.sum()
        root_probabilities = torch.lerp(
            torch.softmax(self.root_logits, dim=0),
            root_probabilities,
            step_size,
        )
        self.root_logits.copy_(root_probabilities.clamp_min(1e-30).log())

        transition_counts = statistics["transition_counts"] + pseudocount
        transition_probabilities = transition_counts / transition_counts.sum(
            dim=-1,
            keepdim=True,
        )
        transition_probabilities = torch.lerp(
            torch.softmax(self.transition_logits, dim=-1),
            transition_probabilities,
            step_size,
        )
        self.transition_logits.copy_(transition_probabilities.clamp_min(1e-30).log())

        if self.leaf_distribution == "categorical":
            category_counts = statistics["leaf_category_counts"] + pseudocount
            category_probabilities = category_counts / category_counts.sum(
                dim=-1,
                keepdim=True,
            )
            category_probabilities = torch.lerp(
                torch.softmax(self.categorical_logits, dim=-1),
                category_probabilities,
                step_size,
            )
            self.categorical_logits.copy_(category_probabilities.clamp_min(1e-30).log())
            return

        if step_size != 1.0:
            raise ValueError(
                "stochastic endpoint EM is only implemented for categorical leaves"
            )

        # Use N(0, 1) as the pseudocount reference moments, matching the fixed
        # p_0 boundary of the time-dependent circuit.
        leaf_denominator = statistics["leaf_counts"] + pseudocount
        endpoint_mean = statistics["leaf_sums"] / leaf_denominator
        endpoint_second_moment = (
            statistics["leaf_square_sums"] + pseudocount
        ) / leaf_denominator
        endpoint_variance = (endpoint_second_moment - endpoint_mean.square()).clamp_min(
            min_std**2
        )
        endpoint_log_scale = 0.5 * endpoint_variance.log()

        endpoint = torch.tensor(
            1.0,
            device=self.mean_coefficients.device,
            dtype=self.mean_coefficients.dtype,
        )
        basis, _ = self._time_basis(endpoint)
        basis_scale = basis / basis.square().sum().clamp_min(1e-12)
        self.mean_coefficients.copy_(endpoint_mean[:, :, None] * basis_scale)
        self.log_scale_coefficients.copy_(endpoint_log_scale[:, :, None] * basis_scale)

    def discrete_log_prob(
        self,
        values: torch.Tensor,
        time: torch.Tensor,
        num_bins: int = 256,
    ) -> torch.Tensor:
        if time.ndim != 0:
            raise ValueError("discrete HCLT likelihood requires a scalar time")
        if self.leaf_distribution == "categorical":
            if num_bins != self.categorical_logits.shape[-1]:
                raise ValueError("num_bins must match the categorical HCLT vocabulary")
            return self._circuit_log_prob(self._leaf_log_prob(values, time))
        mean, _, log_scale, _ = self.leaf_parameters(time)
        if self.leaf_distribution == "gaussian-mixture":
            flat_mean = mean.flatten(1)
            flat_log_scale = log_scale.flatten(1)
            component_log_mass = discretized_gaussian_log_mass(
                values,
                flat_mean,
                flat_log_scale,
                num_bins,
            ).reshape(values.shape[0], self.dim, self.num_latents, -1)
            log_weights = torch.log_softmax(self.leaf_mixture_logits, dim=-1)
            leaf_log_mass = torch.logsumexp(
                log_weights[None] + component_log_mass,
                dim=-1,
            )
            return self._circuit_log_prob(leaf_log_mass)
        leaf_log_mass = discretized_gaussian_log_mass(
            values,
            mean,
            log_scale,
            num_bins,
        )
        return self._circuit_log_prob(leaf_log_mass)

    def velocity(self, values: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        if self.leaf_distribution == "categorical":
            raise RuntimeError(
                "categorical endpoint HCLTs do not define a continuous velocity"
            )
        mean, mean_dt, log_scale, log_scale_dt = self._leaf_component_parameters(time)
        component_log_prob = self._component_log_prob(values, mean, log_scale)
        leaf_log_prob = self._mix_leaf_components(component_log_prob)
        with torch.enable_grad():
            if not leaf_log_prob.requires_grad:
                leaf_log_prob.requires_grad_(True)
            root_log_prob = self._circuit_log_prob(leaf_log_prob)
            flows = torch.autograd.grad(
                root_log_prob.sum(),
                leaf_log_prob,
                create_graph=self.training,
            )[0]
        component_velocity = mean_dt + log_scale_dt * (values[:, :, None, None] - mean)
        if self.leaf_distribution == "gaussian-mixture":
            log_weights = torch.log_softmax(self.leaf_mixture_logits, dim=-1)
            responsibilities = torch.softmax(
                log_weights[None] + component_log_prob,
                dim=-1,
            )
            leaf_velocity = (responsibilities * component_velocity).sum(dim=-1)
        else:
            leaf_velocity = component_velocity.squeeze(-1)
        return (flows * leaf_velocity).sum(dim=-1)

    def _sample_dense_child(
        self,
        edge: int,
        parent_states: torch.Tensor,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        probabilities = torch.softmax(
            self.transition_logits[edge, parent_states], dim=-1
        )
        return torch.multinomial(probabilities, 1, generator=generator).squeeze(1)

    def _sample_monarch_child(
        self,
        edge: int,
        parent_states: torch.Tensor,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        states = parent_states
        for layer in (2, 1, 0):
            blocks = torch.div(states, self.block_size, rounding_mode="floor")
            rows = states.remainder(self.block_size)
            probabilities = torch.softmax(
                self.monarch_logits[edge, layer, blocks, rows], dim=-1
            )
            local_inputs = torch.multinomial(
                probabilities, 1, generator=generator
            ).squeeze(1)
            states = blocks * self.block_size + local_inputs
            if layer > 0:
                states = self.monarch_permutation[states]
        return states

    @torch.no_grad()
    def sample_path(
        self,
        num_samples: int,
        times: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        device = self.root_logits.device
        root_probabilities = torch.softmax(self.root_logits, dim=0)
        latent_states = torch.empty(
            num_samples, self.dim, device=device, dtype=torch.long
        )
        latent_states[:, self.root] = torch.multinomial(
            root_probabilities,
            num_samples,
            replacement=True,
            generator=generator,
        )
        queue = deque([self.root])
        while queue:
            parent = queue.popleft()
            for child in self.tree_children[parent]:
                edge = int(self.edge_by_node[child])
                if self.sum_layer == "dense":
                    states = self._sample_dense_child(
                        edge, latent_states[:, parent], generator
                    )
                else:
                    states = self._sample_monarch_child(
                        edge, latent_states[:, parent], generator
                    )
                latent_states[:, child] = states
                queue.append(child)

        variables = torch.arange(self.dim, device=device)[None]
        if self.leaf_distribution == "categorical":
            if not torch.allclose(times, torch.ones_like(times)):
                raise ValueError("categorical HCLTs can only sample the t=1 endpoint")
            category_probabilities = torch.softmax(
                self.categorical_logits[variables, latent_states],
                dim=-1,
            )
            categories = torch.multinomial(
                category_probabilities.flatten(0, 1),
                1,
                generator=generator,
            ).reshape(num_samples, self.dim)
            endpoint = categories.to(self.root_logits.dtype)
            endpoint = endpoint * (2.0 / (self.categorical_logits.shape[-1] - 1)) - 1.0
            return endpoint[None].expand(len(times), -1, -1)

        component_states = None
        if self.leaf_distribution == "gaussian-mixture":
            component_probabilities = torch.softmax(
                self.leaf_mixture_logits[variables, latent_states],
                dim=-1,
            )
            component_states = torch.multinomial(
                component_probabilities.flatten(0, 1),
                1,
                generator=generator,
            ).reshape(num_samples, self.dim)

        dtype = self.mean_coefficients.dtype
        noise = torch.randn(
            num_samples,
            self.dim,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        path = []
        for time in times:
            mean, _, log_scale, _ = self.leaf_parameters(time)
            if self.leaf_distribution == "gaussian-mixture":
                selected_mean = mean[variables, latent_states, component_states]
                selected_scale = torch.exp(
                    log_scale[variables, latent_states, component_states]
                )
            else:
                selected_mean = mean[variables, latent_states]
                selected_scale = torch.exp(log_scale[variables, latent_states])
            path.append(selected_mean + selected_scale * noise)
        return torch.stack(path)

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        time: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        return self.sample_path(num_samples, time.reshape(1), generator)[0]
