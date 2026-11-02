import itertools
import math
import unittest

import torch

from repc.hclt_pc import TimeDependentHCLTPC


class TimeDependentHCLTPCTest(unittest.TestCase):
    def test_dense_log_prob_matches_latent_enumeration(self):
        torch.manual_seed(3)
        parents = [1, -1, 1]
        model = TimeDependentHCLTPC(
            parents=parents,
            num_latents=2,
            time_order=2,
            sum_layer="dense",
        )
        values = torch.randn(2, 3)
        time = torch.tensor(0.6)
        actual = model.log_prob(values, time)

        leaf = model._leaf_log_prob(values, time)
        root_log_weights = torch.log_softmax(model.root_logits, dim=0)
        edge_by_node = model.edge_by_node
        terms = []
        for states in itertools.product(range(2), repeat=3):
            term = root_log_weights[states[1]]
            for node in (0, 2):
                transition = torch.log_softmax(
                    model.transition_logits[edge_by_node[node]], dim=-1
                )
                term = term + transition[states[1], states[node]]
            term = term + sum(leaf[:, node, states[node]] for node in range(3))
            terms.append(term)
        expected = torch.logsumexp(torch.stack(terms), dim=0)
        torch.testing.assert_close(actual, expected)

    def test_standard_normal_boundary_and_velocity_training(self):
        configurations = (
            ("dense", 3, "polynomial"),
            ("dense", 3, "fourier"),
            ("monarch", 4, "polynomial"),
        )
        for layer, latents, time_conditioning in configurations:
            with self.subTest(
                layer=layer,
                time_conditioning=time_conditioning,
            ):
                torch.manual_seed(0)
                model = TimeDependentHCLTPC(
                    parents=[-1, 0, 1, 1],
                    num_latents=latents,
                    time_order=2,
                    sum_layer=layer,
                    time_conditioning=time_conditioning,
                    time_embedding_dim=8,
                )
                values = torch.randn(3, 4)
                time = torch.tensor(0.0)
                expected = -0.5 * (
                    values.square().sum(dim=1) + 4 * math.log(2.0 * math.pi)
                )
                torch.testing.assert_close(model.log_prob(values, time), expected)

                model.train()
                velocity = model.velocity(values, torch.tensor(0.4))
                self.assertEqual(velocity.shape, values.shape)
                loss = velocity.square().mean()
                loss.backward()
                self.assertTrue(
                    all(
                        parameter.grad is not None
                        and torch.isfinite(parameter.grad).all()
                        for parameter in model.parameters()
                    )
                )

    def test_fourier_basis_derivative_matches_autograd(self):
        torch.manual_seed(5)
        model = TimeDependentHCLTPC(
            parents=[-1, 0],
            num_latents=2,
            time_conditioning="fourier",
            time_embedding_dim=8,
        )
        time = torch.tensor(0.37, requires_grad=True)
        basis, derivative = model._time_basis(time)
        actual = torch.stack(
            [torch.autograd.grad(value, time, retain_graph=True)[0] for value in basis]
        )
        torch.testing.assert_close(actual, derivative)
        zero_basis, _ = model._time_basis(torch.tensor(0.0))
        torch.testing.assert_close(zero_basis, torch.zeros_like(zero_basis))

        wider_model = TimeDependentHCLTPC(
            parents=[-1, 0],
            num_latents=4,
            time_conditioning="fourier",
            time_embedding_dim=16,
        )
        torch.testing.assert_close(
            model.time_frequencies,
            wider_model.time_frequencies[: len(model.time_frequencies)],
        )

    def test_gaussian_mixture_leaf_is_normalized_and_trainable(self):
        torch.manual_seed(7)
        model = TimeDependentHCLTPC(
            parents=[-1, 0],
            num_latents=2,
            time_conditioning="fourier",
            time_embedding_dim=8,
            leaf_distribution="gaussian-mixture",
            leaf_components=2,
        )
        values = torch.randn(5, 2)
        expected = -0.5 * (values.square().sum(dim=1) + 2 * math.log(2.0 * math.pi))
        torch.testing.assert_close(model.log_prob(values, torch.tensor(0.0)), expected)

        normalization_model = TimeDependentHCLTPC(
            parents=[-1],
            num_latents=2,
            time_conditioning="fourier",
            time_embedding_dim=8,
            leaf_distribution="gaussian-mixture",
            leaf_components=2,
        )
        bin_centers = torch.linspace(-1.0, 1.0, 256)[:, None]
        discrete_log_prob = normalization_model.discrete_log_prob(
            bin_centers,
            torch.tensor(0.6),
        )
        torch.testing.assert_close(
            torch.logsumexp(discrete_log_prob, dim=0),
            torch.tensor(0.0),
            atol=1e-5,
            rtol=1e-5,
        )

        model.train()
        velocity = model.velocity(values, torch.tensor(0.4))
        velocity.square().mean().backward()
        self.assertTrue(
            all(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
        )
        path = model.sample_path(5, torch.tensor([0.0, 0.5, 1.0]))
        self.assertEqual(path.shape, (3, 5, 2))





if __name__ == "__main__":
    unittest.main()
