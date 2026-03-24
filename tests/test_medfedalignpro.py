import unittest
from types import SimpleNamespace

import torch

from algorithm.server.medfedalignpro import MedFedAlignProServer


class MedFedAlignProServerTests(unittest.TestCase):
    def test_aggregate_prototypes_uses_weighted_mean_and_ema(self):
        server = MedFedAlignProServer.__new__(MedFedAlignProServer)
        server.classification_model = SimpleNamespace(feature_dim=3)
        server.device = torch.device("cpu")
        server.args = SimpleNamespace(proto_momentum=0.5)
        server.global_anchors = torch.zeros(2, 3)
        server.global_anchor_mask = torch.zeros(2, dtype=torch.bool)

        first_round = [
            {
                "prototypes": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
                "counts": torch.tensor([10.0, 4.0]),
                "reliabilities": torch.tensor([1.0, 1.0]),
            },
            {
                "prototypes": torch.tensor([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0]]),
                "counts": torch.tensor([10.0, 0.0]),
                "reliabilities": torch.tensor([1.0, 0.0]),
            },
        ]
        server.aggregate_prototypes(first_round)
        self.assertTrue(torch.allclose(server.global_anchors[0], torch.tensor([2.0, 0.0, 0.0])))
        self.assertTrue(torch.allclose(server.global_anchors[1], torch.tensor([0.0, 1.0, 0.0])))
        self.assertTrue(bool(server.global_anchor_mask[0]))
        self.assertTrue(bool(server.global_anchor_mask[1]))

        second_round = [
            {
                "prototypes": torch.tensor([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0]]),
                "counts": torch.tensor([10.0, 10.0]),
                "reliabilities": torch.tensor([1.0, 1.0]),
            }
        ]
        server.aggregate_prototypes(second_round)
        self.assertTrue(torch.allclose(server.global_anchors[0], torch.tensor([3.5, 0.0, 0.0])))
        self.assertTrue(torch.allclose(server.global_anchors[1], torch.tensor([0.0, 3.0, 0.0])))


if __name__ == "__main__":
    unittest.main()
