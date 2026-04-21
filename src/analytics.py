"""
Embedding Space Analytics.

This file computes quantitative metrics on the CLIP embedding space:
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
        Start the analytics engine.
        
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



    def cross_event_retrieval_matrix(self) -> np.ndarray:
        """
        Compute a directional cross-event retrieval transfer matrix.

        For each source event i and target event j, computes the mean cosine
        similarity between a sample of source images and the target centroid.
        The diagonal is the mean similarity of source images to their own centroid.

        Returns:
            np.ndarray of shape (num_events, num_events).
            matrix[i, j] = mean_sim(source_event_i_images, centroid_j)
        """
        
        if "cross_event_retrieval" in self._cache:
            return self._cache["cross_event_retrieval"]

        sample_size = config.ANALYTICS_SAMPLE_SIZE

        # Precompute all centroids (L2-normalised)
        centroids = []
        
        for event in self.events:
            embs = self._get_event_embeddings(event)
            c = embs.mean(axis=0)
            c = c / np.linalg.norm(c)
            centroids.append(c)
            
        centroids = np.array(centroids)  # (num_events, 512)
        

        n_events = len(self.events)
        matrix = np.zeros((n_events, n_events), dtype=np.float32)

        for i, src_event in enumerate(self.events):
            embs = self._get_event_embeddings(src_event)
            count = len(embs)
            
            if count > sample_size:
                rng = np.random.RandomState(config.RANDOM_SEED)
                idx = rng.choice(count, size=sample_size, replace=False)
                sampled = embs[idx]
            
            else:
                sampled = embs
            
            # L2-normalised embeddings dot product == cosine similarity
            
            sims = sampled @ centroids.T 
            matrix[i] = sims.mean(axis=0)

        self._cache["cross_event_retrieval"] = matrix
        
        
        return matrix
    
    

    def loo_classification_accuracy(self) -> Dict[str, Any]:
        """
        Leave-one-out classification accuracy across events.

        For each held-out event, classify its sampled images against the
        centroids of the remaining events. An image is correctly classified
        if the nearest remaining centroid belongs to the same disaster type.

        Returns:
            Dict with event_accuracies, type_accuracies, overall_accuracy.
        """
        
        if "loo_classification" in self._cache:
            return self._cache["loo_classification"]

        sample_size = config.ANALYTICS_SAMPLE_SIZE
        type_groups = config.DISASTER_TYPE_GROUPS

        # Precompute all centroids
        centroids = {}
        
        for event in self.events:
            embs = self._get_event_embeddings(event)
            c = embs.mean(axis=0)
            c = c / np.linalg.norm(c)
            centroids[event] = c

        event_accuracies = {}


        for held_out in self.events:
            remaining = [e for e in self.events if e != held_out]
            remaining_centroids = np.array([centroids[e] for e in remaining])

            embs = self._get_event_embeddings(held_out)
            count = len(embs)
        
            if count > sample_size:
                rng = np.random.RandomState(config.RANDOM_SEED)
                idx = rng.choice(count, size=sample_size, replace=False)
                sampled = embs[idx]
        
            else:
                sampled = embs

            sims = sampled @ remaining_centroids.T  # (N_sample, 6)
            nearest_idx = np.argmax(sims, axis=1)

            held_out_type = type_groups.get(held_out, "Unknown")
            
            correct = sum(1 for ni in nearest_idx if type_groups.get(remaining[ni], "Unknown") == held_out_type)
            
            event_accuracies[held_out] = float(correct) / len(nearest_idx)

        # Aggregate by disaster type
        type_sums = {}
        type_counts = {}
        
        
        for event, acc in event_accuracies.items():
            t = type_groups.get(event, "Unknown")
            type_sums[t] = type_sums.get(t, 0.0) + acc
            type_counts[t] = type_counts.get(t, 0) + 1
        
        type_accuracies = {t: type_sums[t] / type_counts[t] for t in type_sums}
        

        overall = float(np.mean(list(event_accuracies.values())))

        result = {
            "event_accuracies": event_accuracies,
            "type_accuracies": type_accuracies,
            "overall_accuracy": overall,
        }
        
        self._cache["loo_classification"] = result
        
        
        return result



    def disaster_type_grouping_analysis(self) -> Dict[str, Any]:
        """
        Within-type vs across-type similarity using centroid matrix.

        Groups the 7 events into 3 disaster types and tests whether CLIP
        representations cluster by disaster type.

        Returns:
            Dict with within_type, overall_within, overall_across, separation_ratio.
        """
        
        if "disaster_type_grouping" in self._cache:
            return self._cache["disaster_type_grouping"]

        type_groups = config.DISASTER_TYPE_GROUPS
        matrix = self.inter_event_similarity_matrix()  # reuse cached
        events = self.events

        event_types = [type_groups.get(e, "Unknown") for e in events]
        type_names = sorted(set(event_types))

        within_type_sims = {t: [] for t in type_names}
        across_type_sims = []
        

        for i in range(len(events)):
            for j in range(i + 1, len(events)):
                sim = float(matrix[i, j])
                
                if event_types[i] == event_types[j]:
                    within_type_sims[event_types[i]].append(sim)
                    
                else:
                    across_type_sims.append(sim)

        within_type = {t: float(np.mean(sims)) if sims else None for t, sims in within_type_sims.items()}
        

        valid_within = [v for v in within_type.values() if v is not None]
        overall_within = float(np.mean(valid_within)) if valid_within else None
        overall_across = float(np.mean(across_type_sims)) if across_type_sims else None
        
        separation_ratio = (
            overall_within / overall_across
            if overall_within is not None and overall_across and overall_across > 0
            else None
        )

        result = {
            "within_type": within_type,
            "overall_within": overall_within,
            "overall_across": overall_across,
            "separation_ratio": separation_ratio,
            "event_types": event_types,
        }
        self._cache["disaster_type_grouping"] = result
        
        return result
    
    

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
        retrieval = self.cross_event_retrieval_matrix()
        loo = self.loo_classification_accuracy()
        grouping = self.disaster_type_grouping_analysis()

        return {
            "global_summary": self.global_summary(),
            "per_event_stats": self.per_event_stats(),
            "inter_event_similarity": {
                "events": self.events,
                "matrix": matrix.tolist(),
            },
            "cross_event_retrieval": {
                "events": self.events,
                "matrix": retrieval.tolist(),
            },
            "loo_classification": loo,
            "disaster_type_grouping": grouping,
        }
