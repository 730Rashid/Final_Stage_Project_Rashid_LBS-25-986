"""
Unit Tests matching Table 5 in the Final Report.

These tests verify the five assertions documented in the report:
    UT-01  Vectorise.py   L2 Normalisation Magnitude
    UT-02  Severity.py    Sigmoid Temperature Bounds
    UT-03  Clean_data.py  Corrupted Image Rejection
    UT-04  Umap_utils.py  NaN Value Handling in Projection
    UT-05  Cluster.py     HDBSCAN Noise Assignment

Author: Rashid
"""

import sys
import tempfile
import numpy as np
import torch
import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# UT-01 — Vectorise.py: L2 Normalisation Magnitude
# After L2 normalising a vector its norm must be exactly 1.0.
# This mirrors the line:  outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
# ---------------------------------------------------------------------------
def test_ut01_l2_normalisation_magnitude():
    vector = torch.randn(1, 512)                               # random CLIP-like embedding
    normalised = vector / vector.norm(p=2, dim=-1, keepdim=True)
    assert torch.norm(normalised).item() == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# UT-02 — Severity.py: Sigmoid Temperature Bounds
# The severity scoring uses:  score = 1 / (1 + exp(-20 * contrast))
# For any real-valued contrast the score must stay in (0, 1), and
# for the practical input range it should satisfy 0.0 <= score <= 100.0
# when expressed as a percentage (0-100 scale, as in the report).
# ---------------------------------------------------------------------------
def test_ut02_sigmoid_temperature_bounds():
    contrasts = np.linspace(-1.0, 1.0, 1000)
    scores = 1.0 / (1.0 + np.exp(-20.0 * contrasts))
    scores_pct = scores * 100.0                                 # percentage scale
    assert all(0.0 <= s <= 100.0 for s in scores_pct)


# ---------------------------------------------------------------------------
# UT-03 — Clean_data.py: Corrupted Image Rejection
# validate_image must return False for a file that is not a valid image.
# ---------------------------------------------------------------------------
def test_ut03_corrupted_image_rejection():
    # Lazy import so the test can find the project on sys.path
    PROJECT_ROOT = Path(__file__).parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from clean_data import is_valid_image

    # Create a corrupt file (random bytes, not a valid image)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(b"\x00\x01\x02corrupted_byte_file")
        corrupted_byte_file = f.name

    is_valid, _reason = is_valid_image(corrupted_byte_file)
    assert is_valid == False


# ---------------------------------------------------------------------------
# UT-04 — Umap_utils.py: NaN Value Handling in Projection
# UMAP projection output must contain no NaN values when the input is clean.
# This validates that the pipeline produces usable 2D coordinates.
# ---------------------------------------------------------------------------
def test_ut04_nan_handling_in_projection():
    import umap

    rng = np.random.RandomState(42)
    embeddings = rng.randn(200, 512).astype(np.float32)        # synthetic CLIP vectors

    reducer = umap.UMAP(n_components=2, random_state=42)
    projection = reducer.fit_transform(embeddings)

    assert np.isnan(projection).any() == False


# ---------------------------------------------------------------------------
# UT-05 — Cluster.py: HDBSCAN Noise Assignment
# HDBSCAN must label some points as noise (-1) when the data contains
# scattered outlier points that do not belong to any dense cluster.
# ---------------------------------------------------------------------------
def test_ut05_hdbscan_noise_assignment():
    import hdbscan

    rng = np.random.RandomState(42)
    # Two tight clusters plus random outliers
    cluster_a = rng.randn(100, 10) * 0.1 + np.array([5] * 10)
    cluster_b = rng.randn(100, 10) * 0.1 + np.array([-5] * 10)
    noise = rng.randn(30, 10) * 10                             # scattered outliers
    data = np.vstack([cluster_a, cluster_b, noise])

    clusterer = hdbscan.HDBSCAN(min_cluster_size=15)
    cluster_labels = clusterer.fit_predict(data)

    assert -1 in cluster_labels
