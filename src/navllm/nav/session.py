"""
Session management for conversational cube navigation.

Following Definition 5: A session Σ = <s_0, ..., s_T> where each state
s_t = <V_t, u_t, a_t> contains the view, utterance, and action.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import json
import uuid

from navllm.cube.view import CubeView
from navllm.cube.actions import NavigationAction
from navllm.llm.scorer import ViewSummary


@dataclass
class SessionState:
    """
    A single state in a navigation session: s_t = <V_t, u_t, a_t>.
    
    Attributes:
        view: Current cube view V_t
        utterance: User utterance u_t
        action: Navigation action a_t that led to next state (None for final state)
        timestamp: When this state was created
        view_summary: Cached view summary for LLM
        recommendations: List of recommended views at this step
        selected_recommendation_idx: Index of recommendation user selected (-1 if manual)
    """
    view: CubeView
    utterance: str
    action: Optional[NavigationAction] = None
    timestamp: datetime = field(default_factory=datetime.now)
    view_summary: Optional[ViewSummary] = None
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    selected_recommendation_idx: int = -1
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize state for logging."""
        return {
            "view_id": self.view.view_id,
            "view_levels": self.view.levels,
            "view_filters": [f.to_dict() for f in self.view.filters],
            "utterance": self.utterance,
            "action": self.action.describe() if self.action else None,
            "timestamp": self.timestamp.isoformat(),
            "recommendations": self.recommendations,
            "selected_recommendation_idx": self.selected_recommendation_idx
        }


@dataclass
class Session:
    """
    A conversational analysis session Σ = <s_0, ..., s_T>.
    
    Tracks the sequence of states, enabling:
    - History-based preference scoring
    - Diversity computation
    - Session logging for evaluation
    """
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    states: List[SessionState] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def current_state(self) -> Optional[SessionState]:
        """Get current (latest) state."""
        return self.states[-1] if self.states else None
    
    @property
    def current_view(self) -> Optional[CubeView]:
        """Get current view V_t."""
        return self.current_state.view if self.current_state else None
    
    @property
    def step_count(self) -> int:
        """Number of navigation steps taken."""
        return len(self.states)
    
    def get_history(self) -> List[Tuple[CubeView, str, ViewSummary]]:
        """
        Get history Hist_t for preference scoring.
        
        Returns list of (view, utterance, summary) tuples.
        """
        history = []
        for state in self.states:
            if state.view_summary:
                history.append((state.view, state.utterance, state.view_summary))
        return history
    
    def get_visited_views(self) -> List[CubeView]:
        """Get all views visited in this session."""
        return [s.view for s in self.states]
    
    def add_state(self, view: CubeView, utterance: str,
                  view_summary: Optional[ViewSummary] = None) -> SessionState:
        """
        Add a new state to the session.
        
        Args:
            view: The new current view
            utterance: User's utterance at this step
            view_summary: Pre-computed view summary
        
        Returns:
            The created SessionState
        """
        state = SessionState(
            view=view,
            utterance=utterance,
            view_summary=view_summary
        )
        self.states.append(state)
        return state
    
    def record_action(self, action: NavigationAction):
        """Record the action taken from current state."""
        if self.current_state:
            self.current_state.action = action
    
    def record_recommendations(self, recommendations: List[Dict[str, Any]],
                                selected_idx: int = -1):
        """Record recommendations shown at current step."""
        if self.current_state:
            self.current_state.recommendations = recommendations
            self.current_state.selected_recommendation_idx = selected_idx
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize session for logging/storage."""
        return {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "step_count": self.step_count,
            "metadata": self.metadata,
            "states": [s.to_dict() for s in self.states]
        }
    
    def save(self, filepath: str):
        """Save session to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str, schema) -> "Session":
        """Load session from JSON file (requires schema for view reconstruction)."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        session = cls(
            session_id=data["session_id"],
            start_time=datetime.fromisoformat(data["start_time"]),
            metadata=data.get("metadata", {})
        )
        
        for state_data in data["states"]:
            view = CubeView.from_dict(schema, {
                "levels": state_data["view_levels"],
                "filters": state_data["view_filters"],
                "aggregations": {}
            })
            state = SessionState(
                view=view,
                utterance=state_data["utterance"],
                timestamp=datetime.fromisoformat(state_data["timestamp"]),
                recommendations=state_data.get("recommendations", []),
                selected_recommendation_idx=state_data.get("selected_recommendation_idx", -1)
            )
            session.states.append(state)
        
        return session


class SessionLogger:
    """
    Logs session data for evaluation and analysis.
    """
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        import os
        os.makedirs(log_dir, exist_ok=True)
    
    def log_session(self, session: Session):
        """Save session to log directory."""
        filepath = f"{self.log_dir}/session_{session.session_id}.json"
        session.save(filepath)
    
    def log_step(self, session: Session, step_data: Dict[str, Any]):
        """Log individual step data."""
        filepath = f"{self.log_dir}/session_{session.session_id}_steps.jsonl"
        with open(filepath, 'a') as f:
            step_data["session_id"] = session.session_id
            step_data["step"] = session.step_count
            f.write(json.dumps(step_data) + "\n")
