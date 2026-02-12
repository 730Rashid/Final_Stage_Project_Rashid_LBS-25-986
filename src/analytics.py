"""
Embedding Space Analytics.

Computes quantitative metrics on the CLIP embedding space:
- Per-event statistics (cohesion, spread)
- Inter-event similarity matrix
- Intra-event similarity distributions
- Global embedding space summary

Author: Rashid
Supervisor: XinHui Ma
Project: Visualising Natural Disaster Image Embeddings
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config.settings import config


class EmbeddingAnalytics:
    """Analyse the CLIP embedding space across disaster events."""

    def __init__(self, embeddings: np.ndarray, df: pd.DataFrame) -> None:
        """
        Initialise the analytics engine.
        
        Args:
            embeddings: np.ndarray of shape (N, 512), L2-normalised CLIP embeddings.
            df: pd.DataFrame with columns including 'event' and 'original_idx'.
        """
        self.embeddings = embeddings
        self.df = df
        self.events = sorted(df["event"].unique())
        self._cache: Dict[str, Any] = {}

    def _get_event_embeddings(self, event: str) -> np.ndarray:
        """Get embeddings for a single event."""
        mask = self.df["event"] == event
        indices = self.df.loc[mask, "original_idx"].values
        return self.embeddings[indices]

    def per_event_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Compute per-event statistics.

        Returns:
            Dict mapping event name to {count, cohesion, spread}.
            Cohesion = mean pairwise cosine similarity (sampled).
            Spread = std of pairwise cosine similarities.
        """
        if "per_event_stats" in self._cache:
            return self._cache["per_event_stats"]

        sample_size = config.ANALYTICS_SAMPLE_SIZE
        stats = {}

        for event in self.events:
            embs = self._get_event_embeddings(event)
            count = len(embs)

            if count < 2:
                stats[event] = {"count": count, "cohesion": 1.0, "spread": 0.0}
                continue

            # Sample if the event has too many images for pairwise computation
            if count > sample_size:
                rng = np.random.RandomState(config.RANDOM_SEED)
                idx = rng.choice(count, size=sample_size, replace=False)
                sampled = embs[idx]
            else:
                sampled = embs

            sim_matrix = cosine_similarity(sampled)
            # Extract upper triangle (excluding diagonal)
            triu_indices = np.triu_indices(len(sampled), k=1)
            pairwise_sims = sim_matrix[triu_indices]

            stats[event] = {
                "count": count,
                "cohesion": float(np.mean(pairwise_sims)),
                "spread": float(np.std(pairwise_sims)),
            }

        self._cache["per_event_stats"] = stats
        return stats

    def inter_event_similarity_matrix(self) -> np.ndarray:
        """
        Compute centroid-to-centroid cosine similarity between all events.

        Returns:
            np.ndarray of shape (num_events, num_events) with cosine similarities.
        """
        if "inter_event_matrix" in self._cache:
            return self._cache["inter_event_matrix"]

        centroids = []
        for event in self.events:
            embs = self._get_event_embeddings(event)
            centroid = embs.mean(axis=0)
            # L2-normalise the centroid
            centroid = centroid / np.linalg.norm(centroid)
            centroids.append(centroid)

        centroids = np.array(centroids)
        matrix = cosine_similarity(centroids)

        self._cache["inter_event_matrix"] = matrix
        return matrix

    def intra_event_distributions(self) -> Dict[str, List[float]]:
        """
        Get sampled pairwise similarity distributions per event for box plots.

        Returns:
            Dict mapping event name to list of float similarity values.
        """
        if "intra_distributions" in self._cache:
            return self._cache["intra_distributions"]

        sample_size = config.ANALYTICS_SAMPLE_SIZE
        distributions = {}

        for event in self.events:
            embs = self._get_event_embeddings(event)
            count = len(embs)

            if count < 2:
                distributions[event] = [1.0]
                continue

            if count > sample_size:
                rng = np.random.RandomState(config.RANDOM_SEED)
                idx = rng.choice(count, size=sample_size, replace=False)
                sampled = embs[idx]
            else:
                sampled = embs

            sim_matrix = cosine_similarity(sampled)
            triu_indices = np.triu_indices(len(sampled), k=1)
            pairwise_sims = sim_matrix[triu_indices]

            # Subsample the pairs for frontend performance
            if len(pairwise_sims) > 2000:
                rng = np.random.RandomState(config.RANDOM_SEED)
                pairwise_sims = rng.choice(pairwise_sims, size=2000, replace=False)

            distributions[event] = [float(s) for s in pairwise_sims]

        self._cache["intra_distributions"] = distributions
        return distributions

    def global_summary(self) -> Dict[str, Any]:
        """
        Compute global embedding space statistics.

        Returns:
            Dict with mean, std, min, max of sampled pairwise similarities.
        """
        if "global_summary" in self._cache:
            return self._cache["global_summary"]

        # Sample from the full dataset
        sample_size = config.ANALYTICS_SAMPLE_SIZE
        n = len(self.embeddings)

        rng = np.random.RandomState(config.RANDOM_SEED)
        idx = rng.choice(n, size=min(sample_size, n), replace=False)
        sampled = self.embeddings[idx]

        sim_matrix = cosine_similarity(sampled)
        triu_indices = np.triu_indices(len(sampled), k=1)
        pairwise_sims = sim_matrix[triu_indices]

        summary = {
            "total_images": n,
            "embedding_dim": self.embeddings.shape[1],
            "num_events": len(self.events),
            "global_mean_similarity": float(np.mean(pairwise_sims)),
            "global_std_similarity": float(np.std(pairwise_sims)),
            "global_min_similarity": float(np.min(pairwise_sims)),
            "global_max_similarity": float(np.max(pairwise_sims)),
        }

        self._cache["global_summary"] = summary
        return summary

    def export_report(self) -> Dict[str, Any]:
        """
        Bundle all analytics into a JSON-serialisable dict.

        Returns:
            Dict with all analytics results.
        """
        matrix = self.inter_event_similarity_matrix()

        return {
            "global_summary": self.global_summary(),
            "per_event_stats": self.per_event_stats(),
            "inter_event_similarity": {
                "events": self.events,
                "matrix": matrix.tolist(),
            },
        }
