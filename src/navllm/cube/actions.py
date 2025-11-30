"""
OLAP Navigation Actions following Definition 4 in Section 3.2.

Navigation actions a: V → V' transform one view into another via:
- Drill-down: decrease level (finer granularity)
- Roll-up: increase level (coarser granularity)  
- Slice: fix one dimension member
- Dice: apply multi-dimension filter
"""

from dataclasses import dataclass, field
from typing import Optional, Any, List
from enum import Enum
from abc import ABC, abstractmethod

from navllm.cube.view import CubeView, Filter
from navllm.cube.schema import CubeSchema


class ActionType(Enum):
    """Types of OLAP navigation actions."""
    DRILL_DOWN = "drill_down"
    ROLL_UP = "roll_up"
    SLICE = "slice"
    DICE = "dice"
    UNSLICE = "unslice"  # Remove a filter


@dataclass
class NavigationAction(ABC):
    """
    Abstract base class for OLAP navigation actions.
    
    An action a: V ⇀ V' is a partial function mapping views to views.
    """
    
    @property
    @abstractmethod
    def action_type(self) -> ActionType:
        """Return the action type."""
        pass
    
    @abstractmethod
    def apply(self, view: CubeView) -> Optional[CubeView]:
        """
        Apply this action to a view, returning the new view.
        Returns None if action is not applicable.
        """
        pass
    
    @abstractmethod
    def describe(self) -> str:
        """Return human-readable description of the action."""
        pass
    
    @abstractmethod
    def is_applicable(self, view: CubeView) -> bool:
        """Check if this action can be applied to the given view."""
        pass


@dataclass
class DrillDownAction(NavigationAction):
    """
    Drill-down on a dimension: move to finer granularity.
    Decreases level index by 1 (e.g., year → quarter → month).
    """
    dimension: str
    
    @property
    def action_type(self) -> ActionType:
        return ActionType.DRILL_DOWN
    
    def is_applicable(self, view: CubeView) -> bool:
        dim = view.schema.get_dimension(self.dimension)
        if dim is None:
            return False
        current_level = view.get_level(self.dimension)
        return dim.can_drill_down(current_level)
    
    def apply(self, view: CubeView) -> Optional[CubeView]:
        if not self.is_applicable(view):
            return None
        
        new_view = view.copy()
        dim = view.schema.get_dimension(self.dimension)
        current_level = view.get_level(self.dimension)
        child_level = dim.get_child_level(current_level)
        new_view.levels[self.dimension] = child_level.name
        return new_view
    
    def describe(self) -> str:
        return f"Drill down on {self.dimension}"


@dataclass
class RollUpAction(NavigationAction):
    """
    Roll-up on a dimension: move to coarser granularity.
    Increases level index by 1 (e.g., month → quarter → year).
    """
    dimension: str
    
    @property
    def action_type(self) -> ActionType:
        return ActionType.ROLL_UP
    
    def is_applicable(self, view: CubeView) -> bool:
        dim = view.schema.get_dimension(self.dimension)
        if dim is None:
            return False
        current_level = view.get_level(self.dimension)
        return dim.can_roll_up(current_level)
    
    def apply(self, view: CubeView) -> Optional[CubeView]:
        if not self.is_applicable(view):
            return None
        
        new_view = view.copy()
        dim = view.schema.get_dimension(self.dimension)
        current_level = view.get_level(self.dimension)
        parent_level = dim.get_parent_level(current_level)
        new_view.levels[self.dimension] = parent_level.name
        
        # Remove filters on finer levels of this dimension
        new_view.filters = [
            f for f in new_view.filters
            if f.dimension != self.dimension
        ]
        return new_view
    
    def describe(self) -> str:
        return f"Roll up on {self.dimension}"


@dataclass
class SliceAction(NavigationAction):
    """
    Slice: fix a single value for a dimension at current level.
    Adds an equality filter (dimension.level = value).
    """
    dimension: str
    value: Any
    
    @property
    def action_type(self) -> ActionType:
        return ActionType.SLICE
    
    def is_applicable(self, view: CubeView) -> bool:
        dim = view.schema.get_dimension(self.dimension)
        if dim is None:
            return False
        # Check if already sliced on this dimension
        for f in view.filters:
            if f.dimension == self.dimension and f.operator == "=":
                return False
        return True
    
    def apply(self, view: CubeView) -> Optional[CubeView]:
        if not self.is_applicable(view):
            return None
        
        new_view = view.copy()
        level_name = view.levels[self.dimension]
        new_filter = Filter(
            dimension=self.dimension,
            level=level_name,
            operator="=",
            value=self.value
        )
        new_view.filters.append(new_filter)
        return new_view
    
    def describe(self) -> str:
        return f"Slice {self.dimension} = {self.value}"


@dataclass
class DiceAction(NavigationAction):
    """
    Dice: apply a range or set filter on a dimension.
    More general than slice (supports IN, BETWEEN, etc.).
    """
    dimension: str
    operator: str
    value: Any
    
    @property
    def action_type(self) -> ActionType:
        return ActionType.DICE
    
    def is_applicable(self, view: CubeView) -> bool:
        dim = view.schema.get_dimension(self.dimension)
        return dim is not None
    
    def apply(self, view: CubeView) -> Optional[CubeView]:
        if not self.is_applicable(view):
            return None
        
        new_view = view.copy()
        level_name = view.levels[self.dimension]
        new_filter = Filter(
            dimension=self.dimension,
            level=level_name,
            operator=self.operator,
            value=self.value
        )
        new_view.filters.append(new_filter)
        return new_view
    
    def describe(self) -> str:
        return f"Dice {self.dimension} {self.operator} {self.value}"


@dataclass
class UnsliceAction(NavigationAction):
    """
    Remove a filter from a dimension (undo slice/dice).
    """
    dimension: str
    
    @property
    def action_type(self) -> ActionType:
        return ActionType.UNSLICE
    
    def is_applicable(self, view: CubeView) -> bool:
        return any(f.dimension == self.dimension for f in view.filters)
    
    def apply(self, view: CubeView) -> Optional[CubeView]:
        if not self.is_applicable(view):
            return None
        
        new_view = view.copy()
        new_view.filters = [
            f for f in new_view.filters if f.dimension != self.dimension
        ]
        return new_view
    
    def describe(self) -> str:
        return f"Remove filter on {self.dimension}"


def generate_candidate_actions(view: CubeView, 
                                member_values: dict = None) -> List[NavigationAction]:
    """
    Generate all applicable navigation actions from current view.
    
    Args:
        view: Current cube view
        member_values: Optional dict of {dimension: [values]} for slice candidates
    
    Returns:
        List of applicable NavigationAction objects
    """
    actions = []
    
    for dim in view.schema.dimensions:
        # Drill-down actions
        drill = DrillDownAction(dimension=dim.name)
        if drill.is_applicable(view):
            actions.append(drill)
        
        # Roll-up actions
        rollup = RollUpAction(dimension=dim.name)
        if rollup.is_applicable(view):
            actions.append(rollup)
        
        # Slice actions (if member values provided)
        if member_values and dim.name in member_values:
            for val in member_values[dim.name]:
                slice_action = SliceAction(dimension=dim.name, value=val)
                if slice_action.is_applicable(view):
                    actions.append(slice_action)
        
        # Unslice actions
        unslice = UnsliceAction(dimension=dim.name)
        if unslice.is_applicable(view):
            actions.append(unslice)
    
    return actions
