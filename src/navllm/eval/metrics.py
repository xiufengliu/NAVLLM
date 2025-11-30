"""
Evaluation metrics for NavLLM.

Implements metrics from Section 7.1.4:
- Hit Rate@B: fraction of tasks where target view reached within B steps
- Steps to First Hit: average steps to reach target
- Cumulative Interestingness: sum of I_data over visited views
- Redundancy: fraction of near-duplicate views
- Coverage: number of distinct pattern clusters visited
"""

from dataclasses import dataclass
from typing import List, Set, Dict, Any, Optional
import numpy as np

from navllm.cube.view import CubeView
from navllm.nav.session import Session


@dataclass
class EvaluationResult:
    """Results from evaluating a single session."""
    session_id: str
    task_id: str
    hit: bool
    steps_to_hit: Optional[int]
    cumulative_i_data: float
    redundancy: float
    coverage: int
    total_steps: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "task_id": self.task_id,
            "hit": self.hit,
            "steps_to_hit": self.steps_to_hit,
            "cumulative_i_data": self.cumulative_i_data,
            "redundancy": self.redundancy,
            "coverage": self.coverage,
            "total_steps": self.total_steps
        }


class MetricsCalculator:
    """
    Calculates evaluation metrics for navigation sessions.
    """
    
    def __init__(self, similarity_threshold: float = 0.8):
        """
        Args:
            similarity_threshold: Threshold for considering views as near-duplicates
        """
        self.similarity_threshold = similarity_threshold
    
    def evaluate_session(self,
                         session: Session,
                         target_views: List[CubeView],
                         i_data_scores: List[float],
                         task_id: str = "") -> EvaluationResult:
        """
        Evaluate a single navigation session.
        
        Args:
            session: The navigation session to evaluate
            target_views: Set of target views (ground truth)
            i_data_scores: I_data scores for each visited view
            task_id: Identifier for the task
        
        Returns:
            EvaluationResult with all metrics
        """
        visited_views = session.get_visited_views()
        
        # Hit Rate and Steps to Hit
        hit = False
        steps_to_hit = None
        
        for i, view in enumerate(visited_views):
            if self._is_target_hit(view, target_views):
                hit = True
                steps_to_hit = i + 1
                break
        
        # Cumulative Interestingness
        cumulative_i_data = sum(i_data_scores)
        
        # Redundancy
        redundancy = self._compute_redundancy(visited_views)
        
        # Coverage
        coverage = self._compute_coverage(visited_views)
        
        return EvaluationResult(
            session_id=session.session_id,
            task_id=task_id,
            hit=hit,
            steps_to_hit=steps_to_hit,
            cumulative_i_data=cumulative_i_data,
            redundancy=redundancy,
            coverage=coverage,
            total_steps=len(visited_views)
        )
    
    def _is_target_hit(self, view: CubeView, targets: List[CubeView]) -> bool:
        """Check if view matches any target view."""
        for target in targets:
            if self._views_match(view, target):
                return True
        return False
    
    def _views_match(self, v1: CubeView, v2: CubeView) -> bool:
        """Check if two views are equivalent."""
        # Check levels
        if v1.levels != v2.levels:
            return False
        
        # Check filters (order-independent)
        f1 = {(f.dimension, f.level, f.operator, str(f.value)) for f in v1.filters}
        f2 = {(f.dimension, f.level, f.operator, str(f.value)) for f in v2.filters}
        
        return f1 == f2
    
    def _compute_redundancy(self, views: List[CubeView]) -> float:
        """
        Compute redundancy: fraction of near-duplicate views.
        
        A view is redundant if it's very similar to a previously visited view.
        """
        if len(views) <= 1:
            return 0.0
        
        redundant_count = 0
        
        for i, view in enumerate(views[1:], 1):
            for prev_view in views[:i]:
                if self._view_similarity(view, prev_view) > self.similarity_threshold:
                    redundant_count += 1
                    break
        
        return redundant_count / (len(views) - 1)
    
    def _view_similarity(self, v1: CubeView, v2: CubeView) -> float:
        """Compute similarity between two views (0 to 1)."""
        # Level similarity
        level_matches = sum(1 for d in v1.levels if v1.levels[d] == v2.levels.get(d))
        level_sim = level_matches / len(v1.levels) if v1.levels else 1.0
        
        # Filter similarity (Jaccard)
        f1 = {(f.dimension, f.level, f.operator, str(f.value)) for f in v1.filters}
        f2 = {(f.dimension, f.level, f.operator, str(f.value)) for f in v2.filters}
        
        if not f1 and not f2:
            filter_sim = 1.0
        elif not f1 or not f2:
            filter_sim = 0.0
        else:
            intersection = len(f1 & f2)
            union = len(f1 | f2)
            filter_sim = intersection / union
        
        return 0.7 * level_sim + 0.3 * filter_sim
    
    def _compute_coverage(self, views: List[CubeView]) -> int:
        """
        Compute coverage: number of distinct pattern clusters.
        
        Clusters views by their dimension level combinations.
        """
        clusters = set()
        
        for view in views:
            # Create cluster key from levels
            cluster_key = tuple(sorted(view.levels.items()))
            clusters.add(cluster_key)
        
        return len(clusters)
    
    def aggregate_results(self, results: List[EvaluationResult]) -> Dict[str, float]:
        """
        Aggregate metrics across multiple sessions.
        
        Returns:
            Dict with mean and std for each metric
        """
        if not results:
            return {}
        
        hits = [r.hit for r in results]
        steps = [r.steps_to_hit for r in results if r.steps_to_hit is not None]
        i_data = [r.cumulative_i_data for r in results]
        redundancy = [r.redundancy for r in results]
        coverage = [r.coverage for r in results]
        
        return {
            "hit_rate": np.mean(hits),
            "hit_rate_std": np.std(hits),
            "mean_steps_to_hit": np.mean(steps) if steps else None,
            "steps_to_hit_std": np.std(steps) if steps else None,
            "mean_cumulative_i_data": np.mean(i_data),
            "cumulative_i_data_std": np.std(i_data),
            "mean_redundancy": np.mean(redundancy),
            "redundancy_std": np.std(redundancy),
            "mean_coverage": np.mean(coverage),
            "coverage_std": np.std(coverage),
            "n_sessions": len(results)
        }
