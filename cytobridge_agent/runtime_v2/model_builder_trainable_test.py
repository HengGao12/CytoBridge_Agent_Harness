import unittest

import torch

from CytoBridge.tl.trainer import TrainingPipeline
from CytoBridge.tl.training_algorithm import TrainingDataBundle


class _CustomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.velocity_net = torch.nn.Linear(3, 2)
        self.growth_net = torch.nn.Linear(3, 1)
        self.birth_head = torch.nn.Linear(3, 1)
        self.cytobridge_component_modules = {"growth": ["birth_head"]}


class ModelBuilderTrainableTest(unittest.TestCase):
    def test_component_module_from_model_builder_is_trainable(self):
        bundle = TrainingDataBundle(
            adata=None,
            time_points=[0.0, 1.0],
            latent_by_time=[torch.randn(4, 2), torch.randn(4, 2)],
            obs_indices_by_time=[list(range(4)), list(range(4))],
        )
        config = {
            "model": {"components": ["velocity", "growth"]},
            "training": {
                "defaults": {"sigma": 0.0},
                "plan": [
                    {
                        "name": "fm",
                        "mode": "flow_matching",
                        "lr": 1e-3,
                        "train_strategy": "g",
                    }
                ],
            },
            "ckpt_dir": "/tmp/cytobridge_model_builder_trainable_test",
        }
        model = _CustomModel()
        pipeline = TrainingPipeline(
            model,
            config,
            batch_size=2,
            device=torch.device("cpu"),
            training_data=bundle,
        )

        pipeline._setup_stage(config["training"]["plan"][0])

        self.assertFalse(any(p.requires_grad for p in model.velocity_net.parameters()))
        self.assertTrue(any(p.requires_grad for p in model.growth_net.parameters()))
        self.assertTrue(any(p.requires_grad for p in model.birth_head.parameters()))
        optimizer_params = {
            name
            for name, param in model.named_parameters()
            if any(
                param is selected
                for group in pipeline.optimizer.param_groups
                for selected in group["params"]
            )
        }
        self.assertIn("growth_net.weight", optimizer_params)
        self.assertIn("birth_head.weight", optimizer_params)
        self.assertNotIn("velocity_net.weight", optimizer_params)


if __name__ == "__main__":
    unittest.main()
