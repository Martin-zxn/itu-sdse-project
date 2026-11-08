import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import torch

from modeling.model import SimpleCNN, load_model
from modeling.train import save_checkpoint
from monitoring.drift import detect_drift


class ModelTests(unittest.TestCase):
    def test_output_shape_and_checkpoint_reload(self):
        torch.manual_seed(7)
        model = SimpleCNN().eval()
        inputs = torch.randn(2, 3, 64, 64)
        expected = model(inputs)
        self.assertEqual(tuple(expected.shape), (2, 1))

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "vial_cnn_model.pth"
            save_checkpoint(model, checkpoint_path, threshold=0.5)
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(
                set(checkpoint),
                {
                    "model_state_dict",
                    "architecture",
                    "input_shape",
                    "output_shape",
                    "threshold",
                    "num_classes",
                },
            )
            self.assertEqual(checkpoint["architecture"], "SimpleCNN")
            self.assertEqual(checkpoint["input_shape"], (3, 64, 64))
            self.assertEqual(checkpoint["output_shape"], (1,))

            reloaded = load_model(checkpoint_path)
            actual = reloaded(inputs)
            self.assertEqual(tuple(actual.shape), (2, 1))
            torch.testing.assert_close(actual, expected)


class DriftTests(unittest.TestCase):
    def test_no_logs_returns_notebook_response_keys(self):
        result = detect_drift([], baseline_stats={})
        self.assertEqual(
            set(result),
            {
                "overall_drift_detected",
                "num_samples",
                "confidence_drift",
                "class_distribution_drift",
                "error",
            },
        )
        self.assertFalse(result["overall_drift_detected"])

    def test_logged_requests_return_notebook_response_keys(self):
        timestamp = datetime.now(timezone.utc).isoformat()
        logs = [
            {"timestamp": timestamp, "confidence": 0.2, "prediction": 0},
            {"timestamp": timestamp, "confidence": 0.8, "prediction": 1},
        ]
        baseline = {
            "confidence_statistics": {"mean": 0.5, "std": 0.2},
            "class_distribution": {
                "class_0_count": 1,
                "class_1_count": 1,
                "total": 2,
            },
        }
        result = detect_drift(logs, baseline_stats=baseline)
        self.assertEqual(
            set(result),
            {
                "overall_drift_detected",
                "num_samples",
                "confidence_drift",
                "class_distribution_drift",
                "confidence_stats",
                "class_distribution",
            },
        )
        self.assertEqual(result["num_samples"], 2)
        self.assertEqual(
            set(result["confidence_drift"]),
            {
                "ks_statistic",
                "ks_pvalue",
                "mean_diff",
                "current_mean",
                "baseline_mean",
                "drift_detected",
                "threshold_ks",
                "threshold_mean",
            },
        )
        self.assertEqual(
            set(result["class_distribution_drift"]),
            {"ks_statistic", "ks_pvalue", "drift_detected", "threshold_ks"},
        )


if __name__ == "__main__":
    unittest.main()
