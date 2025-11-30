"""
Preference Scorer: Computes I_pref(V | Hist_t) using LLM.

Following Section 5.3 and Section 6.2 of the paper:
- Constructs view summaries for LLM consumption
- Encodes conversational history
- Parses and validates LLM scores
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

from navllm.llm.client import LLMClient
from navllm.cube.view import CubeView
from navllm.cube.engine import ViewResult

logger = logging.getLogger(__name__)


@dataclass
class ViewSummary:
    """
    Schema-level summary of a view for LLM consumption.
    
    Contains only descriptions and pre-computed statistics,
    never raw data.
    """
    view_id: str
    description: str
    grouping: Dict[str, str]
    filters: List[Dict[str, Any]]
    statistics: Dict[str, Any]
    
    def to_text(self) -> str:
        """Convert to text format for LLM prompt."""
        parts = [self.description]
        
        if self.statistics:
            stats_parts = []
            for measure, stats in self.statistics.items():
                if isinstance(stats, dict):
                    stats_parts.append(
                        f"{measure}: mean={stats.get('mean', 'N/A'):.2f}, "
                        f"CV={stats.get('cv', 0):.2f}, "
                        f"outliers={stats.get('outlier_count', 0)}, "
                        f"deviation={stats.get('deviation_category', 'N/A')}"
                    )
            if stats_parts:
                parts.append(f"[Stats: {'; '.join(stats_parts)}]")
        
        return " ".join(parts)


class PreferenceScorer:
    """
    Computes preference-based interestingness I_pref(V | Hist_t).
    
    Uses LLM to score candidate views based on conversational context.
    """
    
    SCORING_PROMPT_TEMPLATE = """You are an assistant helping an analyst explore a multidimensional data cube. Based on the conversation history below, rate how relevant each candidate view is to the analyst's current analysis goal.

Conversation history:
{history}

Current view: {current_view}
User's latest request: "{current_utterance}"

Candidate views to evaluate:
{candidates}

For each candidate, provide a relevance score from 0.0 to 1.0 where 1.0 means highly relevant to the user's stated interest. Consider dimension alignment, filter relevance, and analytical value.

Output JSON only: {{"scores": [0.X, 0.Y, ...]}}"""

    def __init__(self, llm_client: LLMClient, 
                 max_history_steps: int = 5,
                 fallback_score: float = 0.5):
        """
        Initialize preference scorer.
        
        Args:
            llm_client: LLM client for API calls
            max_history_steps: Max recent steps to include in full detail
            fallback_score: Score to use when LLM fails
        """
        self.llm_client = llm_client
        self.max_history_steps = max_history_steps
        self.fallback_score = fallback_score
    
    def create_view_summary(self, view: CubeView, 
                            result: ViewResult,
                            target_measure: str) -> ViewSummary:
        """
        Create schema-level summary of a view.
        
        All numeric values are pre-computed by cube engine.
        """
        # Build description
        desc_parts = []
        
        # Grouping description
        group_parts = []
        for dim, level in view.levels.items():
            if level != "All":
                group_parts.append(f"{dim}.{level}")
        if group_parts:
            desc_parts.append(f"Grouped by [{', '.join(group_parts)}]")
        else:
            desc_parts.append("Total aggregate (no grouping)")
        
        # Filter description
        if view.filters:
            filter_parts = [
                f"{f.dimension}.{f.level} {f.operator} {f.value}"
                for f in view.filters
            ]
            desc_parts.append(f"Filtered: {', '.join(filter_parts)}")
        
        desc_parts.append(f"Returns {result.row_count} rows")
        
        # Get statistics from cube engine
        stats = {}
        if target_measure in result.statistics:
            measure_stats = result.statistics[target_measure]
            mean = measure_stats.get("mean", 0)
            std = measure_stats.get("std", 0)
            cv = std / (abs(mean) + 1e-6) if mean != 0 else 0
            
            stats[target_measure] = {
                "mean": mean,
                "std": std,
                "cv": cv,
                "min": measure_stats.get("min", 0),
                "max": measure_stats.get("max", 0),
                "outlier_count": sum(1 for v in result.data[target_measure] 
                                     if abs(v - mean) > 2 * std) if std > 0 else 0,
                "deviation_category": "high" if cv > 0.3 else ("medium" if cv > 0.1 else "low")
            }
        
        return ViewSummary(
            view_id=view.view_id,
            description="; ".join(desc_parts),
            grouping=view.levels,
            filters=[f.to_dict() for f in view.filters],
            statistics=stats
        )
    
    def encode_history(self, history: List[Tuple[CubeView, str, ViewSummary]]) -> str:
        """
        Encode conversational history for LLM prompt.
        
        Args:
            history: List of (view, utterance, summary) tuples
        
        Returns:
            Formatted history string
        """
        if not history:
            return "(No prior history)"
        
        lines = []
        n = len(history)
        
        for i, (view, utterance, summary) in enumerate(history):
            step_num = i + 1
            
            # Full detail for recent steps, compressed for older
            if i >= n - self.max_history_steps:
                lines.append(f"[Step {step_num}] View: {summary.to_text()}")
                lines.append(f"         User said: \"{utterance}\"")
            else:
                # Compressed: just utterance and high-level view info
                lines.append(f"[Step {step_num}] {summary.description} | User: \"{utterance}\"")
        
        return "\n".join(lines)
    
    def score_candidates(self, 
                         candidates: List[Tuple[CubeView, ViewSummary]],
                         current_view_summary: ViewSummary,
                         current_utterance: str,
                         history: List[Tuple[CubeView, str, ViewSummary]]) -> List[float]:
        """
        Score candidate views using LLM.
        
        Args:
            candidates: List of (view, summary) tuples
            current_view_summary: Summary of current view
            current_utterance: User's latest utterance
            history: Conversational history
        
        Returns:
            List of preference scores in [0, 1]
        """
        if not candidates:
            return []
        
        # Build prompt
        history_text = self.encode_history(history)
        
        candidate_lines = []
        for i, (view, summary) in enumerate(candidates, 1):
            candidate_lines.append(f"{i}. {summary.to_text()}")
        
        prompt = self.SCORING_PROMPT_TEMPLATE.format(
            history=history_text,
            current_view=current_view_summary.to_text(),
            current_utterance=current_utterance,
            candidates="\n".join(candidate_lines)
        )
        
        # Call LLM
        response = self.llm_client.complete_json(prompt)
        
        # Parse and validate scores
        scores = response.get("scores", [])
        
        if not scores or len(scores) != len(candidates):
            logger.warning(
                f"Invalid LLM response: expected {len(candidates)} scores, "
                f"got {len(scores)}. Using fallback."
            )
            return [self.fallback_score] * len(candidates)
        
        # Validate score range
        validated_scores = []
        for s in scores:
            try:
                score = float(s)
                score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
                validated_scores.append(score)
            except (TypeError, ValueError):
                validated_scores.append(self.fallback_score)
        
        return validated_scores


class ExplanationGenerator:
    """
    Generates natural language explanations for recommended views.
    """
    
    EXPLANATION_PROMPT_TEMPLATE = """You are an assistant helping an analyst explore a data cube. Generate a brief, factual explanation for why the following view is recommended.

Current analysis context:
- User's goal: "{utterance}"
- Current view: {current_view}

Recommended view: {recommended_view}
- Data interestingness score: {i_data:.2f} (higher = more anomalous)
- Preference score: {i_pref:.2f} (higher = more aligned with user intent)

Generate a 1-2 sentence explanation that:
1. Explains what this view shows
2. Why it might be interesting given the user's stated goal
3. References specific statistics if relevant

Be factual and concise. Do not invent statistics not provided above."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
    
    def generate_explanation(self,
                             recommended_summary: ViewSummary,
                             current_summary: ViewSummary,
                             utterance: str,
                             i_data: float,
                             i_pref: float) -> str:
        """Generate explanation for a recommended view."""
        prompt = self.EXPLANATION_PROMPT_TEMPLATE.format(
            utterance=utterance,
            current_view=current_summary.to_text(),
            recommended_view=recommended_summary.to_text(),
            i_data=i_data,
            i_pref=i_pref
        )
        
        response = self.llm_client.complete(prompt, max_tokens=150)
        
        if response.success:
            return response.content.strip()
        else:
            # Fallback explanation
            return f"This view {recommended_summary.description.lower()} and may reveal patterns related to your query."
