"""
Multi Model Embedding Comparison (Ablation Study).

Computes dimensionality reduction, clustering, and evaluation metrics
across 3 models x 3 reductions x 2 clustering algorithms:

  Models:     CLIP ViT B/32, SigLIP base, ResNet50 (ImageNet)
  Reductions: UMAP, t SNE, PCA
  Clustering: HDBSCAN, K Means (k=7)
  Metrics:    Silhouette, Davies Bouldin, Calinski Harabasz,
              Trustworthiness, Continuity

Results are cached to disk as JSON for the Dash app's Model Comparison tab.

"""

import sys
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.manifold import trustworthiness, TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import umap
import hdbscan

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import config
from utils.event_utils import parse_event

_SCATTER_SAMPLE = 2000
_TRUST_SAMPLE   = 3000


def load_all_embeddings() -> Dict[str, np.ndarray]:
    """Load embeddings for all three models from disk."""
    embeddings = {}

    for model_key in config.COMPARISON_MODELS:
        path = config.COMPARISON_DIR / "{}_embeddings.npy".format(model_key)

        if not path.exists():
            raise FileNotFoundError(
                "Missing embeddings: {}. Run vectorise_models.py first.".format(path))

        embs = np.load(path)
        embeddings[model_key] = embs
        print("Loaded {} embeddings: {}".format(model_key, embs.shape))

    return embeddings


def load_event_labels() -> List[str]:
    """Load CLIP filenames and extract event labels."""
    fn_path = config.COMPARISON_DIR / "clip_filenames.json"

    with open(fn_path, "r") as f:
        filenames = json.load(f)

    return [parse_event(p) for p in filenames]


def run_reduction(embeddings: np.ndarray, method: str) -> np.ndarray:
    """Apply dimensionality reduction (UMAP, t SNE, or PCA) to 2D."""
    n, d = embeddings.shape

    if method == "umap":
        reducer = umap.UMAP(
            n_neighbors=config.UMAP_N_NEIGHBOURS,
            min_dist=config.UMAP_MIN_DIST,
            n_components=config.UMAP_N_COMPONENTS,
            metric=config.UMAP_METRIC,
            random_state=config.UMAP_RANDOM_STATE,
            verbose=True,
        )

        return reducer.fit_transform(embeddings)

    elif method == "tsne":
        print("t SNE: {} samples x {} dim ...".format(n, d))

        # L2 normalise so Euclidean distance is monotonically equivalent
        # to cosine distance. Using metric="cosine" in sklearn t SNE
        # forces a full pairwise distance matrix which runs out of memory
        # at 17k samples, while euclidean uses ball tree kNN.
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        emb_normed = embeddings / norms

        reducer = TSNE(
            n_components=2,
            perplexity=config.TSNE_PERPLEXITY,
            learning_rate=config.TSNE_LEARNING_RATE,
            max_iter=config.TSNE_N_ITER,
            random_state=config.TSNE_RANDOM_STATE,
            metric="euclidean",
            verbose=2,
        )

        return reducer.fit_transform(emb_normed)

    elif method == "pca":
        reducer = PCA(
            n_components=config.PCA_N_COMPONENTS,
            whiten=config.PCA_WHITEN,
            random_state=config.RANDOM_SEED,
        )

        return reducer.fit_transform(embeddings)

    raise ValueError("Unknown reduction method: {}".format(method))


def run_clustering(embeddings: np.ndarray, method: str) -> np.ndarray:
    """Apply HDBSCAN or K Means clustering on high dimensional embeddings."""

    if method == "hdbscan":
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=config.HDBSCAN_MIN_CLUSTER_SIZE,
            min_samples=config.HDBSCAN_MIN_SAMPLES,
            metric=config.HDBSCAN_METRIC,
            cluster_selection_method=config.HDBSCAN_CLUSTER_SELECTION_METHOD,
            core_dist_n_jobs=-1,
        )

        return clusterer.fit_predict(embeddings)

    elif method == "kmeans":
        clusterer = KMeans(
            n_clusters=config.KMEANS_N_CLUSTERS,
            random_state=config.RANDOM_SEED,
            n_init=10,
        )

        return clusterer.fit_predict(embeddings)

    raise ValueError("Unknown clustering method: {}".format(method))


def _subsample_pair(high_dim: np.ndarray, low_dim: np.ndarray,max_n: int = _TRUST_SAMPLE) -> tuple:
    """Return a consistent subsample of both arrays if N exceeds max_n."""
    
    n = len(high_dim)

    if n <= max_n:
        return high_dim, low_dim, n

    rng = np.random.RandomState(42)
    idx = rng.choice(n, max_n, replace=False)

    return high_dim[idx], low_dim[idx], max_n


def compute_trustworthiness(high_dim: np.ndarray, low_dim: np.ndarray, n_neighbors: int = 12) -> float:
    """Trustworthiness: are low dim neighbours also near in the original space?"""
    n_orig = len(high_dim)
    hd, ld, n = _subsample_pair(high_dim, low_dim)

    if n < n_orig:
        print("Computing trustworthiness (subsampled {}/{} samples, k={})...".format(
            n, n_orig, n_neighbors))
    else:
        print("Computing trustworthiness ({} samples, k={})...".format(n, n_neighbors))

    print("Building distance matrices...", end=" ", flush=True)
    t0 = time.time()
    
    score = float(trustworthiness(hd, ld, n_neighbors=n_neighbors))
    print("done ({:.1f}s) -> {:.4f}".format(time.time() - t0, score))

    return score


def compute_continuity(high_dim: np.ndarray, low_dim: np.ndarray,
                       n_neighbors: int = 12) -> float:
    """Continuity: are original space neighbours preserved in the reduction?"""
    n_orig = len(high_dim)
    hd, ld, n = _subsample_pair(high_dim, low_dim)

    if n < n_orig:
        print("Computing continuity (subsampled {}/{} samples, k={})...".format(
            n, n_orig, n_neighbors))
    else:
        print("Computing continuity ({} samples, k={})...".format(n, n_neighbors))

    print("Building distance matrices", end=" ", flush=True)
    t0 = time.time()
    score = float(trustworthiness(ld, hd, n_neighbors=n_neighbors))
    print("done ({:.1f}s) -> {:.4f}".format(time.time() - t0, score))

    return score


def evaluate_reduction(high_dim: np.ndarray, low_dim: np.ndarray,
                       event_labels: List[str]) -> Dict[str, float]:
    """Evaluate a dimensionality reduction with all 5 metrics."""
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(event_labels)

    print("    Silhouette Score...", end=" ", flush=True)
    t0 = time.time()
    sil = float(silhouette_score(low_dim, encoded))
    print("{:.4f} ({:.1f}s)".format(sil, time.time() - t0))

    print("    Davies Bouldin Index...", end=" ", flush=True)
    t0 = time.time()
    db = float(davies_bouldin_score(low_dim, encoded))
    print("{:.4f} ({:.1f}s)".format(db, time.time() - t0))

    print("    Calinski Harabasz Index...", end=" ", flush=True)
    t0 = time.time()
    ch = float(calinski_harabasz_score(low_dim, encoded))
    print("{:.1f} ({:.1f}s)".format(ch, time.time() - t0))

    trust = compute_trustworthiness(high_dim, low_dim)
    cont  = compute_continuity(high_dim, low_dim)

    return {
        "silhouette_score":       sil,
        "davies_bouldin_index":   db,
        "calinski_harabasz_index": ch,
        "trustworthiness":        trust,
        "continuity":             cont,
    }


def evaluate_clustering(coords: np.ndarray, cluster_labels: np.ndarray) -> Dict[str, Any]:
    """Evaluate clustering quality on 2D coordinates."""
    valid_mask = cluster_labels >= 0
    noise_pct  = float(100.0 * (1.0 - valid_mask.mean()))

    fallback = {
        "silhouette_score":        0.0,
        "davies_bouldin_index":    float("inf"),
        "calinski_harabasz_index": 0.0,
        "noise_pct":               noise_pct,
    }

    if int(valid_mask.sum()) < 2:
        return {**fallback, "n_clusters": 0}

    valid_coords = coords[valid_mask]
    valid_labels = cluster_labels[valid_mask]
    n_unique     = len(set(valid_labels.tolist()))

    if n_unique < 2:
        return {**fallback, "n_clusters": n_unique}

    return {
        "silhouette_score":        float(silhouette_score(valid_coords, valid_labels)),
        "davies_bouldin_index":    float(davies_bouldin_score(valid_coords, valid_labels)),
        "calinski_harabasz_index": float(calinski_harabasz_score(valid_coords, valid_labels)),
        "n_clusters":              n_unique,
        "noise_pct":               noise_pct,
    }


def run_full_comparison() -> Dict[str, Any]:
    """Execute the full 3x3x2 comparison and return all results."""
    print("Loading embeddings...")
    all_embeddings = load_all_embeddings()
    event_labels   = load_event_labels()
    print("  Event labels: {} unique events".format(len(set(event_labels))))
    print()

    results: Dict[str, Any] = {
        "generated_at":      datetime.now().isoformat(),
        "models":            {},
        "reduction_metrics": {},
        "clustering_metrics": {},
        "scatter_data":      {},
    }

    total_steps = len(config.COMPARISON_MODELS) * (
        len(config.COMPARISON_REDUCTIONS) + len(config.COMPARISON_CLUSTERERS))
    step = 0

    for model_key in config.COMPARISON_MODELS:
        embs       = all_embeddings[model_key]
        model_info = config.MODEL_REGISTRY[model_key]

        results["models"][model_key] = {
            "name":          model_info["name"],
            "embedding_dim": int(embs.shape[1]),
            "n_samples":     int(embs.shape[0]),
        }

        results["reduction_metrics"][model_key]  = {}
        results["scatter_data"][model_key]        = {}
        results["clustering_metrics"][model_key]  = {}

        umap_coords_cache = None

        # Reductions
        for reduction in config.COMPARISON_REDUCTIONS:
            step += 1
            print("=" * 60)
            print("[{}/{}] {} + {} reduction".format(
                step, total_steps, model_info["name"], reduction.upper()))
            print("=" * 60)

            t_start = time.time()
            coords  = run_reduction(embs, reduction)
            t_reduction = time.time() - t_start
            print("  Reduction took {:.1f}s".format(t_reduction))

            coords_path = config.COMPARISON_DIR / "{}_{}_coords.npy".format(
                model_key, reduction)
            np.save(coords_path, coords)

            if reduction == "umap":
                umap_coords_cache = coords

            print("  Evaluating reduction metrics...")
            metrics = evaluate_reduction(embs, coords, event_labels)
            metrics["reduction_time_s"] = round(t_reduction, 1)
            results["reduction_metrics"][model_key][reduction] = metrics

            # Subsample for scatter plot data
            n   = len(coords)
            rng = np.random.RandomState(config.RANDOM_SEED)
            idx = rng.choice(n, min(_SCATTER_SAMPLE, n), replace=False)

            results["scatter_data"][model_key][reduction] = {
                "x":      coords[idx, 0].tolist(),
                "y":      coords[idx, 1].tolist(),
                "events": [event_labels[i] for i in idx],
            }

            print()

        # Clustering
        for clusterer in config.COMPARISON_CLUSTERERS:
            step += 1
            print("=" * 60)
            print("[{}/{}] {} + {} clustering".format(
                step, total_steps, model_info["name"], clusterer.upper()))
            print("=" * 60)

            t_start = time.time()
            cluster_labels = run_clustering(embs, clusterer)
            t_cluster = time.time() - t_start
            print("  Clustering took {:.1f}s".format(t_cluster))

            if umap_coords_cache is not None:
                cluster_metrics = evaluate_clustering(umap_coords_cache, cluster_labels)
            else:
                umap_path = config.COMPARISON_DIR / "{}_umap_coords.npy".format(model_key)
                cluster_metrics = evaluate_clustering(np.load(umap_path), cluster_labels)

            cluster_metrics["clustering_time_s"] = round(t_cluster, 1)
            results["clustering_metrics"][model_key][clusterer] = cluster_metrics

            print("  Silhouette: {:.4f} | Clusters: {} | Noise: {:.1f}%".format(
                cluster_metrics["silhouette_score"],
                cluster_metrics["n_clusters"],
                cluster_metrics["noise_pct"],
            ))
            print()

    return results


def main() -> None:
    """Main entry point. Run the full comparison and cache results."""
    print("=" * 60)
    print("Multi Model Embedding Comparison (Ablation Study)")
    print("Timestamp: {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    print("Models:     {}".format(", ".join(config.COMPARISON_MODELS)))
    print("Reductions: {}".format(", ".join(config.COMPARISON_REDUCTIONS)))
    print("Clusterers: {}".format(", ".join(config.COMPARISON_CLUSTERERS)))
    print("=" * 60)
    print()

    t_total = time.time()
    results = run_full_comparison()

    cache_path = config.COMPARISON_CACHE_PATH
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    with open(cache_path, "w") as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - t_total
    print("=" * 60)
    print("Comparison complete in {:.0f}m {:.0f}s".format(elapsed // 60, elapsed % 60))
    print("Results saved to: {}".format(cache_path))
    print("=" * 60)


if __name__ == "__main__":
    main()
