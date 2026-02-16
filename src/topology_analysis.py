"""
Topological & Geometric Analysis of the CLIP Embedding Space.

Phase 1 — Persistent Homology
    Computes the Vietoris-Rips persistence diagram on a 2,000-image
    stratified sample using cosine distance.
    H0 captures how many connected semantic components form (and how
    long they persist), while H1 captures loops / cycles in the space —
    regions of semantic ambiguity not represented by any single cluster.

Phase 2 — Ollivier-Ricci Curvature
    Builds a k-NN cosine-similarity graph on a 500-image stratified
    sample, then computes the exact Ollivier-Ricci curvature (ORC) for
    every edge using the Earth Mover's Distance (Python Optimal Transport).
    Positive ORC  →  locally sphere-like, dense cluster.
    Negative ORC  →  locally tree-like, semantic bridge / bottleneck.

Both results are cached to disk so they are computed only once per
server session.

Author: Rashid
Supervisor: XinHui Ma
Project: Visualising Natural Disaster Image Embeddings
"""

import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple

import ripser
import ot
import networkx as nx

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config.settings import config

# Disk-cache paths
_TOPO_CACHE_PATH = config.VISUALISATION_DIR / "topology_cache.json"

# Sample sizes (chosen to balance accuracy vs. runtime on MX450 / 16 GB RAM)
_PH_SAMPLE_SIZE  = 2000   # persistence homology
_ORC_SAMPLE_SIZE = 500    # Ollivier-Ricci curvature
_KNN_K           = 10     # neighbours in the Ricci k-NN graph


# Helpers

def _stratified_sample(
    embeddings: np.ndarray,
    df: pd.DataFrame,
    events: List[str],
    n: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return a stratified random sample of *n* embeddings (equal per event).

    Returns
    -------
    sample_embs  : (n, D) float32 array
    sample_events: (n,)   object array of event names
    """
    rng = np.random.RandomState(config.RANDOM_SEED)
    per_event = max(1, n // len(events))

    sampled_indices: List[int] = []
    sampled_events:  List[str] = []

    for event in events:
        mask        = df["event"] == event
        event_idx   = df.loc[mask, "original_idx"].values
        k           = min(per_event, len(event_idx))
        chosen      = rng.choice(event_idx, size=k, replace=False)
        sampled_indices.extend(chosen.tolist())
        sampled_events.extend([event] * k)

    embs = embeddings[sampled_indices].astype(np.float32)
    
    
    return embs, np.array(sampled_events)


def _cosine_distance_matrix(embs: np.ndarray) -> np.ndarray:
    """
    Compute a full pairwise cosine-distance matrix.

    L2-normalises embeddings first so the formula cosine_dist = 1 - dot(u, v)
    is exact. Clipped to [0, 2] for numerical safety.
    """
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)   # guard against zero vectors
    normed = embs / norms
    dot = normed @ normed.T
    dist = np.clip(1.0 - dot, 0.0, 2.0)
    return dist.astype(np.float64)



# Main class


class TopologyAnalytics:
    """
    Topological and geometric analysis of the CLIP embedding space.

    Parameters
    ----------
    embeddings : (N, 512) L2-normalised CLIP embeddings
    df         : DataFrame with columns 'event' and 'original_idx'
    """

    def __init__(self, embeddings: np.ndarray, df: pd.DataFrame) -> None:
        self.embeddings = embeddings
        self.df         = df
        self.events     = sorted(df["event"].unique().tolist())
        self._cache: Dict[str, Any] = {}

        # Try to load disk cache
        self._load_disk_cache()

    
    # Disk caching

    def _load_disk_cache(self) -> None:
        """Load previously computed results from disk (if they exist)."""
        if _TOPO_CACHE_PATH.exists():
            try:
                with open(_TOPO_CACHE_PATH, "r", encoding="utf-8") as fh:
                    self._cache = json.load(fh)
            except Exception:
                self._cache = {}

    def _save_disk_cache(self) -> None:
        """Persist current in-memory cache to disk as JSON."""
        try:
            _TOPO_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(_TOPO_CACHE_PATH, "w", encoding="utf-8") as fh:
                json.dump(self._cache, fh)
        except Exception:
            pass   # non-fatal — results already in memory

    
    # Phase 1 — Persistent Homology


    def persistence_homology(self) -> Dict[str, Any]:
        """
        Compute H0 and H1 persistence diagrams on a 2,000-image sample.

        The Vietoris-Rips filtration is built over cosine distances.

        H0 features represent connected components of the semantic space.
        A feature (birth, death) means a new cluster forms at radius
        *birth* and merges with a larger one at radius *death*.  High
        persistence (death − birth) indicates a robust semantic cluster.

        H1 features represent 1-cycles (loops) in the semantic space.
        Persistent H1 features indicate regions of semantic ambiguity
        — areas the embedding space "goes around" rather than through.

        Returns
        -------
        Dict with keys:
            sample_size, h0 (birth/death/persistence), h1 (same)
        """
        if "persistence" in self._cache:
            return self._cache["persistence"]

        sample_embs, _ = _stratified_sample(
            self.embeddings, self.df, self.events, _PH_SAMPLE_SIZE
        )

        # Cosine distance matrix (precomputed for ripser)
        dist_matrix = _cosine_distance_matrix(sample_embs)

        # Ripser: Vietoris-Rips up to dimension 1 (H0 + H1)
        result = ripser.ripser(dist_matrix, distance_matrix=True, maxdim=1)
        dgms   = result["dgms"]

        def _extract(dgm: np.ndarray) -> Dict[str, Any]:
            """Extract finite features from one persistence diagram."""
            finite  = dgm[dgm[:, 1] != np.inf]
            birth   = finite[:, 0].tolist()
            death   = finite[:, 1].tolist()
            persist = (finite[:, 1] - finite[:, 0]).tolist()
            return {
                "birth":           birth,
                "death":           death,
                "persistence":     persist,
                "n_features":      len(finite),
                "max_persistence": float(max(persist)) if persist else 0.0,
                "mean_persistence": float(np.mean(persist)) if persist else 0.0,
            }

        output = {
            "sample_size": len(sample_embs),
            "h0": _extract(dgms[0]),
            "h1": _extract(dgms[1]),
        }

        self._cache["persistence"] = output
        self._save_disk_cache()
        return output

    
    # Phase 2 — Ollivier-Ricci Curvature


    def ollivier_ricci_curvature(self) -> Dict[str, Any]:
        """
        Compute exact Ollivier-Ricci curvature (ORC) for every edge in
        the 10-NN cosine-similarity graph built on a 500-image sample.

        For each edge (u, v):
            μ_u = uniform distribution over the k nearest neighbours of u
            μ_v = uniform distribution over the k nearest neighbours of v
            W1  = Earth Mover's Distance between μ_u and μ_v
                  (exact, computed with Python Optimal Transport)
            κ(u,v) = 1 − W1 / d(u, v)

        Interpretation
        --------------
        κ > 0  →  sphere-like region: the neighbours of u and v overlap
                  heavily → dense, well-formed semantic cluster.
        κ < 0  →  hyperbolic / tree-like region: the neighbourhoods
                  diverge → u and v are a "bridge" between two clusters.
        κ = 0  →  flat (Euclidean-like) neighbourhood.

        Returns
        -------
        Dict with:
            edge_curvatures    : list of float (one per undirected edge)
            event_mean_curvature : {event: mean ORC of intra-event edges}
            global_mean/std/min/max
            pct_positive / pct_negative
            n_positive / n_negative / n_zero
            sample_size, k_neighbors
        """
        if "ricci" in self._cache:
            return self._cache["ricci"]

        sample_embs, sample_events = _stratified_sample(
            self.embeddings, self.df, self.events, _ORC_SAMPLE_SIZE
        )
        n = len(sample_embs)

        cos_dist = _cosine_distance_matrix(sample_embs)  # (n, n)

        # Build undirected k-NN adjacency (union of both directions)
        neighbors: Dict[int, List[int]] = {}
        for i in range(n):
            dists_i   = cos_dist[i].copy()
            dists_i[i] = np.inf               # exclude self
            nn        = np.argsort(dists_i)[:_KNN_K]
            neighbors[i] = nn.tolist()

        # Collect undirected edges
        edge_set: set = set()
        for i in range(n):
            for j in neighbors[i]:
                edge_set.add((min(i, j), max(i, j)))

        # Compute ORC for every edge via Earth Mover's Distance
        edge_curvatures: List[float] = []
        edge_list       = sorted(edge_set)

        for (u, v) in edge_list:
            nu = neighbors[u]
            nv = neighbors[v]

            if not nu or not nv:
                edge_curvatures.append(0.0)
                continue

            d_uv = float(cos_dist[u, v])
            if d_uv < 1e-10:
                edge_curvatures.append(1.0)   # identical vectors → max curvature
                continue

            # Source / target mass distributions (uniform)
            a = np.ones(len(nu), dtype=np.float64) / len(nu)
            b = np.ones(len(nv), dtype=np.float64) / len(nv)

            # Transport cost matrix: pairwise cosine distances between neighbourhoods
            M = cos_dist[np.ix_(nu, nv)]

            try:
                w1    = ot.emd2(a, b, M.copy())
                kappa = 1.0 - w1 / d_uv
            except Exception:
                kappa = 0.0

            edge_curvatures.append(float(np.clip(kappa, -5.0, 5.0)))

        curvatures = np.array(edge_curvatures, dtype=np.float64)

        # Per-event mean curvature (intra-event edges only)
        event_curv_sums:   Dict[str, float] = {e: 0.0 for e in self.events}
        event_curv_counts: Dict[str, int]   = {e: 0   for e in self.events}

        for idx, (u, v) in enumerate(edge_list):
            eu = sample_events[u]
            ev = sample_events[v]
            if eu == ev:
                event_curv_sums[eu]   += edge_curvatures[idx]
                event_curv_counts[eu] += 1

        event_mean_curvature: Dict[str, float] = {}
        for e in self.events:
            cnt = event_curv_counts[e]
            event_mean_curvature[e] = (
                event_curv_sums[e] / cnt if cnt > 0 else 0.0
            )

        output: Dict[str, Any] = {
            "edge_curvatures":      edge_curvatures,
            "event_mean_curvature": event_mean_curvature,
            "global_mean":  float(np.mean(curvatures)),
            "global_std":   float(np.std(curvatures)),
            "global_min":   float(np.min(curvatures)),
            "global_max":   float(np.max(curvatures)),
            "n_positive":   int(np.sum(curvatures > 0)),
            "n_negative":   int(np.sum(curvatures < 0)),
            "n_zero":       int(np.sum(curvatures == 0)),
            "pct_positive": float(np.mean(curvatures > 0) * 100),
            "pct_negative": float(np.mean(curvatures < 0) * 100),
            "sample_size":  n,
            "k_neighbors":  _KNN_K,
            "n_edges":      len(edge_list),
        }

        self._cache["ricci"] = output
        self._save_disk_cache()
        
        
        return output
