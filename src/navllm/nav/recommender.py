"""
NavLLM Recommender: Core recommendation engine.

Implements Algorithm 1 (LLM-Assisted Next-View Recommendation) from Section 5.5:
1. Generate candidate views via OLAP actions
2. Compute I_data (data-driven interestingness)
3. Compute I_pref (LLM preference score)
4. Compute I_div (diversity score)
5. Rank by utility U = λ_data·I_data + λ_pref·I_pref + λ_div·I_div
6. Return top-k recommendations with explanations
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

from navllm.cube.schema import CubeSchema
from navllm.cube.view import CubeView
from navllm.cube.actions import (
    NavigationAction, generate_candidate_actions,
    DrillDownAction, RollUpAction, SliceAction
)
from navllm.cube.engine import CubeEngine, ViewResult
from navllm.llm.client import LLMClient
from navllm.llm.scorer import PreferenceScorer, ExplanationGenerator, ViewSummary
from navllm.nav.session import Session

logger = logging.getLogger(__name__)


@dataclass
class Recommendation:
    """A single view recommendation with scores and explanation."""
    view: CubeView
    action: NavigationAction
    i_data: float
    i_pref: float
    i_div: float
    utility: float
    explanation: str
    summary: ViewSummary
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "view_id": self.view.view_id,
            "action": self.action.describe(),
            "i_data": self.i_data,
            "i_pref": self.i_pref,
            "i_div": self.i_div,
            "utility": self.utility,
            "explanation": self.explanation,
            "description": self.summary.description
        }


@dataclass
class RecommenderConfig:
    """Configuration for the recommender."""
    lambda_data: float = 0.4
    lambda_pref: float = 0.4
    lambda_div: float = 0.2
    candidate_budget: int = 30
    top_k: int = 3
    min_result_rows: int = 3
    generate_explanations: bool = True


class NavLLMRecommender:
    """
    Main recommendation engine for NavLLM.
    
    Combines cube engine (data computations) with LLM (preference scoring)
    to recommend next views during conversational navigation.
    """
    
    def __init__(self, 
                 schema: CubeSchema,
                 cube_engine: CubeEngine,
                 llm_client: LLMClient,
                 config: RecommenderConfig = None):
        """
        Initialize recommender.
        
        Args:
            schema: Cube schema
            cube_engine: Engine for OLAP computations
            llm_client: Client for LLM API calls
            config: Recommender configuration
        """
        self.schema = schema
        self.cube_engine = cube_engine
        self.llm_client = llm_client
        self.config = config or RecommenderConfig()
        
        self.preference_scorer = PreferenceScorer(llm_client)
        self.explanation_generator = ExplanationGenerator(llm_client)
    
    def recommend(self,
                  session: Session,
                  utterance: str,
                  target_measure: str) -> List[Recommendation]:
        """
        Generate top-k view recommendations.
        
        Implements Algorithm 1 from the paper.
        
        Args:
            session: Current navigation session with history
            utterance: User's current utterance
            target_measure: Target measure m* for interestingness
        
        Returns:
            List of top-k Recommendation objects
        """
        current_view = session.current_view
        if current_view is None:
            raise ValueError("Session has no current view")
        
        # Step 1: Generate candidate views
        candidates = self._generate_candidates(current_view)
        logger.info(f"Generated {len(candidates)} candidate views")
        
        if not candidates:
            return []
        
        # Get current view result for parent statistics
        current_result = self.cube_engine.materialize_view(current_view)
        current_summary = self.preference_scorer.create_view_summary(
            current_view, current_result, target_measure
        )
        
        # Step 2-4: Compute scores for each candidate
        scored_candidates = []
        candidate_summaries = []
        
        for action, candidate_view in candidates:
            # Materialize candidate view
            result = self.cube_engine.materialize_view(candidate_view)
            
            # Skip views with too few rows
            if result.row_count < self.config.min_result_rows:
                continue
            
            # Compute I_data (data-driven interestingness)
            i_data = self.cube_engine.compute_data_interestingness(
                candidate_view, current_result, target_measure
            )
            
            # Create summary for LLM
            summary = self.preference_scorer.create_view_summary(
                candidate_view, result, target_measure
            )
            
            # Compute I_div (diversity)
            i_div = self._compute_diversity(candidate_view, session)
            
            scored_candidates.append({
                "action": action,
                "view": candidate_view,
                "result": result,
                "summary": summary,
                "i_data": i_data,
                "i_div": i_div
            })
            candidate_summaries.append((candidate_view, summary))
        
        if not scored_candidates:
            return []
        
        # Step 3: Compute I_pref via LLM (batch call)
        history = session.get_history()
        pref_scores = self.preference_scorer.score_candidates(
            candidate_summaries,
            current_summary,
            utterance,
            history
        )
        
        # Assign preference scores
        for i, candidate in enumerate(scored_candidates):
            candidate["i_pref"] = pref_scores[i] if i < len(pref_scores) else 0.5
        
        # Step 5: Compute utility and rank
        for candidate in scored_candidates:
            candidate["utility"] = (
                self.config.lambda_data * self._normalize(candidate["i_data"]) +
                self.config.lambda_pref * candidate["i_pref"] +
                self.config.lambda_div * self._normalize(candidate["i_div"])
            )
        
        # Sort by utility (descending)
        scored_candidates.sort(key=lambda x: x["utility"], reverse=True)
        
        # Step 6: Select top-k and generate explanations
        top_k = scored_candidates[:self.config.top_k]
        recommendations = []
        
        for candidate in top_k:
            # Generate explanation
            if self.config.generate_explanations:
                explanation = self.explanation_generator.generate_explanation(
                    candidate["summary"],
                    current_summary,
                    utterance,
                    candidate["i_data"],
                    candidate["i_pref"]
                )
            else:
                explanation = candidate["summary"].description
            
            rec = Recommendation(
                view=candidate["view"],
                action=candidate["action"],
                i_data=candidate["i_data"],
                i_pref=candidate["i_pref"],
                i_div=candidate["i_div"],
                utility=candidate["utility"],
                explanation=explanation,
                summary=candidate["summary"]
            )
            recommendations.append(rec)
        
        return recommendations
    
    def _generate_candidates(self, 
                             current_view: CubeView) -> List[Tuple[NavigationAction, CubeView]]:
        """
        Generate candidate next views via OLAP actions.
        
        Applies pruning heuristics from Section 5.1:
        - Exclude undo operations
        - Limit total candidates to budget
        """
        candidates = []
        
        # Get member values for slice candidates
        member_values = {}
        for dim in self.schema.dimensions:
            level_name = current_view.levels[dim.name]
            if level_name != "All":
                members = self.cube_engine.get_dimension_members(
                    dim.name, level_name, current_view.filters
                )
                # Limit slice candidates per dimension
                member_values[dim.name] = members[:5]
        
        # Generate all applicable actions
        actions = generate_candidate_actions(current_view, member_values)
        
        for action in actions:
            new_view = action.apply(current_view)
            if new_view is not None:
                candidates.append((action, new_view))
        
        # Apply budget limit
        if len(candidates) > self.config.candidate_budget:
            # Prioritize: drill-down > slice > roll-up
            def priority(item):
                action = item[0]
                if isinstance(action, DrillDownAction):
                    return 0
                elif isinstance(action, SliceAction):
                    return 1
                elif isinstance(action, RollUpAction):
                    return 2
                return 3
            
            candidates.sort(key=priority)
            candidates = candidates[:self.config.candidate_budget]
        
        return candidates
    
    def _compute_diversity(self, view: CubeView, session: Session) -> float:
        """
        Compute diversity score I_div(V | Hist_t).
        
        Uses feature-based dissimilarity: minimum distance to visited views.
        """
        visited = session.get_visited_views()
        if not visited:
            return 1.0
        
        min_distance = float('inf')
        for visited_view in visited:
            dist = self._view_distance(view, visited_view)
            min_distance = min(min_distance, dist)
        
        # Normalize to [0, 1] range
        max_possible = len(self.schema.dimensions) * 2 + 5  # Rough upper bound
        return min(min_distance / max_possible, 1.0)
    
    def _view_distance(self, v1: CubeView, v2: CubeView) -> float:
        """
        Compute distance between two views.
        
        Based on differences in levels and filters.
        """
        distance = 0.0
        
        # Level differences
        for dim in self.schema.dimensions:
            l1 = v1.levels.get(dim.name)
            l2 = v2.levels.get(dim.name)
            if l1 != l2:
                distance += 1.0
        
        # Filter differences
        f1_set = {(f.dimension, f.level, f.operator, str(f.value)) for f in v1.filters}
        f2_set = {(f.dimension, f.level, f.operator, str(f.value)) for f in v2.filters}
        distance += len(f1_set.symmetric_difference(f2_set)) * 0.5
        
        return distance
    
    def _normalize(self, value: float, max_val: float = 2.0) -> float:
        """Normalize value to [0, 1] range."""
        return min(value / max_val, 1.0)
