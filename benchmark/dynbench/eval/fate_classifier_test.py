from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from benchmark.dynbench.eval.fate_classifier import load_fate_classifier


class FateClassifierCompatibilityTest(unittest.TestCase):
    def test_public_classifier_loads_and_predicts(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "task_packages"
            / "S_balanced_easy_01_seed42"
            / "fate_classifier.pkl"
        )
        if not path.exists():
            self.skipTest("public DynBench classifier fixture is unavailable")
        classifier = load_fate_classifier(str(path))
        n_features = int(classifier.n_features_in_)
        prediction = classifier.predict_proba(np.zeros((1, n_features), dtype=np.float32))
        self.assertEqual(prediction.shape[0], 1)
        self.assertEqual(prediction.shape[1], len(classifier.classes_))


if __name__ == "__main__":
    unittest.main()
