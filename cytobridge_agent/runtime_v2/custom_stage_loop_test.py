import tempfile
import unittest

import torch

from CytoBridge.tl.custom_stage_loop import CustomStageLossResult, run_custom_stage_loop
from CytoBridge.tl.training_algorithm import StageRunnerContext, TrainingDataBundle


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.velocity_net = torch.nn.Linear(1, 1)
        self.extra_head = torch.nn.Linear(1, 1)
        self.cytobridge_component_modules = {"velocity": ["extra_head"]}


class _ToyObjective:
    def build_state(self, context):
        return {"target": torch.ones(4, 1, device=context.device)}

    def sample_batch(self, context, state, epoch):
        del epoch
        return state["target"]

    def compute_loss(self, context, state, batch, epoch):
        del state, epoch
        pred = context.model.velocity_net(batch) + context.model.extra_head(batch)
        loss = torch.mean((pred - batch) ** 2)
        return CustomStageLossResult(loss=loss, logs={"toy_loss": float(loss.detach().cpu())})


class CustomStageLoopTests(unittest.TestCase):
    def test_custom_stage_loop_updates_model_and_writes_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _ToyModel()
            before = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bundle = TrainingDataBundle(
                adata=None,
                time_points=[0.0, 1.0],
                latent_by_time=[torch.zeros(2, 1), torch.ones(2, 1)],
                obs_indices_by_time=[list(range(2)), list(range(2, 4))],
            )
            context = StageRunnerContext(
                model=model,
                config={"ckpt_dir": tmpdir, "model": {"components": ["velocity"]}},
                stage_params={
                    "name": "toy",
                    "mode": "custom",
                    "epochs": 3,
                    "lr": 0.05,
                    "train_strategy": "v",
                    "save_strategy": "last",
                },
                stage_index=0,
                training_data=bundle,
                device=torch.device("cpu"),
                batch_size=4,
                output_dir=tmpdir,
            )

            result = run_custom_stage_loop(context, _ToyObjective())

            self.assertEqual(result.stage_summary["completed_epochs"], 3)
            self.assertEqual(result.stage_summary["runner"], "run_custom_stage_loop")
            self.assertTrue(result.artifacts["checkpoint"].endswith("last_model.pth"))
            self.assertTrue(torch.load(result.artifacts["checkpoint"], map_location="cpu"))
            self.assertFalse(torch.equal(before["velocity_net.weight"], model.state_dict()["velocity_net.weight"]))
            self.assertFalse(torch.equal(before["extra_head.weight"], model.state_dict()["extra_head.weight"]))

    def test_custom_stage_loop_preview_only_does_not_train(self):
        class PreviewObjective(_ToyObjective):
            def evaluate_preview(self, context, state):
                del context, state
                return {"logs": {"preview": True}}

        with tempfile.TemporaryDirectory() as tmpdir:
            model = _ToyModel()
            before = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bundle = TrainingDataBundle(
                adata=None,
                time_points=[0.0, 1.0],
                latent_by_time=[torch.zeros(2, 1), torch.ones(2, 1)],
                obs_indices_by_time=[list(range(2)), list(range(2, 4))],
            )
            context = StageRunnerContext(
                model=model,
                config={"ckpt_dir": tmpdir, "model": {"components": ["velocity"]}},
                stage_params={"name": "toy", "mode": "custom", "epochs": 3, "lr": 0.05},
                stage_index=0,
                training_data=bundle,
                device=torch.device("cpu"),
                batch_size=4,
                output_dir=tmpdir,
                preview_only=True,
            )

            result = run_custom_stage_loop(context, PreviewObjective())

            self.assertTrue(result.logs["preview"])
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(before[key], value))


if __name__ == "__main__":
    unittest.main()
