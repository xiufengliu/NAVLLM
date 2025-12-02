"""
Differentiated baseline navigators - each with distinct behavior.
"""
import numpy as np
from navllm.cube.actions import DrillDownAction, RollUpAction
from navllm.cube.view import CubeView

class HeuristicNav:
    """Pure I_data ranking - picks highest interestingness."""
    def __init__(self, schema, engine):
        self.schema = schema
        self.engine = engine
    
    def select_next(self, session, utterance, measure):
        current = session.current_view
        actions = self._get_drilldown_actions(current)
        if not actions:
            return current, None
        
        parent = self.engine.materialize_view(current)
        best_score, best_action = -1, actions[0]
        for action in actions:
            cand = action.apply(current)
            if cand:
                score = self.engine.compute_data_interestingness(cand, parent, measure)
                if score > best_score:
                    best_score, best_action = score, action
        return best_action.apply(current), best_action
    
    def _get_drilldown_actions(self, view):
        actions = []
        for dim_name, level_name in view.levels.items():
            dim = self.schema.get_dimension(dim_name)
            level = dim.get_level_by_name(level_name)
            if level.order > 0:
                actions.append(DrillDownAction(dim_name))
        return actions


class SeeDBNav:
    """SeeDB-style: I_data * sqrt(n) for statistical significance weighting."""
    def __init__(self, schema, engine):
        self.schema = schema
        self.engine = engine
    
    def select_next(self, session, utterance, measure):
        current = session.current_view
        actions = self._get_drilldown_actions(current)
        if not actions:
            return current, None
        
        parent = self.engine.materialize_view(current)
        best_score, best_action = -1, actions[0]
        for action in actions:
            cand = action.apply(current)
            if cand:
                result = self.engine.materialize_view(cand)
                i_data = self.engine.compute_data_interestingness(cand, parent, measure)
                # SeeDB weighting: more rows = more reliable
                score = i_data * np.sqrt(min(result.row_count, 100))
                if score > best_score:
                    best_score, best_action = score, action
        return best_action.apply(current), best_action
    
    def _get_drilldown_actions(self, view):
        actions = []
        for dim_name, level_name in view.levels.items():
            dim = self.schema.get_dimension(dim_name)
            level = dim.get_level_by_name(level_name)
            if level.order > 0:
                actions.append(DrillDownAction(dim_name))
        return actions


class GreedyBestNav:
    """Greedy: picks dimension with highest variance in child level."""
    def __init__(self, schema, engine):
        self.schema = schema
        self.engine = engine
    
    def select_next(self, session, utterance, measure):
        current = session.current_view
        actions = self._get_drilldown_actions(current)
        if not actions:
            return current, None
        
        best_var, best_action = -1, actions[0]
        for action in actions:
            cand = action.apply(current)
            if cand:
                result = self.engine.materialize_view(cand)
                if result.row_count > 1 and measure in result.statistics:
                    var = result.statistics[measure].get('std', 0) ** 2
                else:
                    var = 0
                if var > best_var:
                    best_var, best_action = var, action
        return best_action.apply(current), best_action
    
    def _get_drilldown_actions(self, view):
        actions = []
        for dim_name, level_name in view.levels.items():
            dim = self.schema.get_dimension(dim_name)
            level = dim.get_level_by_name(level_name)
            if level.order > 0:
                actions.append(DrillDownAction(dim_name))
        return actions


class BFSNav:
    """BFS: cycles through dimensions in fixed order."""
    def __init__(self, schema, engine):
        self.schema = schema
        self.engine = engine
        self.dim_order = [d.name for d in schema.dimensions]
        self.idx = 0
    
    def select_next(self, session, utterance, measure):
        current = session.current_view
        actions = self._get_drilldown_actions(current)
        if not actions:
            return current, None
        
        # Try dimensions in order
        for _ in range(len(self.dim_order)):
            dim = self.dim_order[self.idx]
            self.idx = (self.idx + 1) % len(self.dim_order)
            for a in actions:
                if a.dimension == dim:
                    return a.apply(current), a
        return actions[0].apply(current), actions[0]
    
    def _get_drilldown_actions(self, view):
        actions = []
        for dim_name, level_name in view.levels.items():
            dim = self.schema.get_dimension(dim_name)
            level = dim.get_level_by_name(level_name)
            if level.order > 0:
                actions.append(DrillDownAction(dim_name))
        return actions


class RandomNav:
    """Random: uniform random selection."""
    def __init__(self, schema, engine):
        self.schema = schema
        self.engine = engine
    
    def select_next(self, session, utterance, measure):
        current = session.current_view
        actions = self._get_drilldown_actions(current)
        if not actions:
            return current, None
        action = np.random.choice(actions)
        return action.apply(current), action
    
    def _get_drilldown_actions(self, view):
        actions = []
        for dim_name, level_name in view.levels.items():
            dim = self.schema.get_dimension(dim_name)
            level = dim.get_level_by_name(level_name)
            if level.order > 0:
                actions.append(DrillDownAction(dim_name))
        return actions
