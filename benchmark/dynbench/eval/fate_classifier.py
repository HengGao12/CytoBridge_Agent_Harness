"""
Unified MLP fate classifier for DynBench v2.

Trained on training data with known fate labels.
Used consistently across all metrics that need fate assignment:
  - GT generation (BoolODE simulation endpoints)
  - M3: classify predicted holdout cells
  - M4: classify SDE forward simulation endpoints  
  - M5: classify perturbation simulation endpoints
"""

import numpy as np
import pickle
from typing import Optional


class _InferenceOnlyRandomState:
    """Accept incompatible NumPy RNG pickle state during inference-only loads."""

    def __setstate__(self, state):
        self.state = state


def _compat_randomstate_ctor(*args, **kwargs):
    return _InferenceOnlyRandomState()


def _compat_bit_generator_ctor(*args, **kwargs):
    return _InferenceOnlyRandomState()


class _NumpyCompatUnpickler(pickle.Unpickler):
    """Load sklearn predictors serialized by a newer NumPy release.

    DynBench uses the classifier for ``predict``/``predict_proba`` only. Its
    serialized RNG is training state and is not consulted for inference, so it
    is safe to replace an unsupported RNG constructor after a normal load has
    failed.
    """

    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = "numpy.core" + module[len("numpy._core"):]
        if module == "numpy.random._pickle":
            if name == "__randomstate_ctor":
                return _compat_randomstate_ctor
            if name == "__bit_generator_ctor":
                return _compat_bit_generator_ctor
        return super().find_class(module, name)


def _build_mlp():
    """Build a simple MLP for fate classification using sklearn."""
    from sklearn.neural_network import MLPClassifier
    return MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        max_iter=500,
        early_stopping=False,
        random_state=42,
        verbose=False,
    )


def _concat_time(expr: np.ndarray, time: np.ndarray) -> np.ndarray:
    """Concatenate expression matrix with time column."""
    time_col = np.asarray(time, dtype=np.float32).reshape(-1, 1)
    return np.hstack([expr, time_col])


def train_fate_classifier(
    expr: np.ndarray,
    labels: np.ndarray,
    time: np.ndarray,
    save_path: Optional[str] = None,
) -> object:
    """
    Train MLP fate classifier on ALL training data (including Progenitor).
    
    Features = gene expression + time (so the classifier can distinguish
    early progenitors from late transitioning cells at similar expression).
    
    Args:
        expr: (n_cells, n_genes) gene expression
        labels: (n_cells,) fate labels (e.g. 'Fate_A', 'Fate_B', 'Progenitor')
        time: (n_cells,) time values for each cell
        save_path: optional path to save the trained model
        
    Returns:
        Trained MLPClassifier
    """
    # Filter out only unknown/invalid labels, keep ALL biological categories
    valid_mask = np.array([l not in ['unknown', ''] for l in labels])
    
    if valid_mask.sum() < 10:
        raise ValueError(f"Too few valid cells ({valid_mask.sum()}) to train classifier")
    
    train_features = _concat_time(expr[valid_mask], time[valid_mask])
    train_labels = labels[valid_mask]
    
    clf = _build_mlp()
    clf.fit(train_features, train_labels)
    
    # Report accuracy
    train_acc = clf.score(train_features, train_labels)
    unique_labels, counts = np.unique(train_labels, return_counts=True)
    label_dist = dict(zip(unique_labels, counts))
    print(f"  Fate classifier trained: {valid_mask.sum()} cells, "
          f"features=expr({expr.shape[1]})+time, "
          f"accuracy={train_acc:.3f}, classes={list(clf.classes_)}")
    print(f"  Label distribution: {label_dist}")
    
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(clf, f)
        print(f"  Saved to: {save_path}")
    
    return clf


def load_fate_classifier(path: str) -> object:
    """Load a trained fate classifier across supported NumPy pickle versions."""
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except (ModuleNotFoundError, AttributeError, TypeError, ValueError) as exc:
        message = str(exc)
        is_numpy_compat_error = (
            "numpy._core" in message
            or "BitGenerator" in message
            or "numpy.random" in message
        )
        if not is_numpy_compat_error:
            raise
        with open(path, 'rb') as f:
            return _NumpyCompatUnpickler(f).load()


def classify_cells(
    clf,
    expr: np.ndarray,
    time: np.ndarray,
) -> np.ndarray:
    """
    Classify cells into fates using trained MLP.
    
    Args:
        clf: trained classifier
        expr: (n_cells, n_genes) gene expression
        time: (n_cells,) time values
        
    Returns:
        (n_cells,) predicted fate labels
    """
    features = _concat_time(expr, time)
    return clf.predict(features)
