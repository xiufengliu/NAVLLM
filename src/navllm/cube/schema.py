"""
Cube Schema definitions following the formal model in Section 3.1.

Definition 1 (Cube Schema): S = <D, M> where D is dimensions with hierarchies,
and M is the set of measures.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum


class AggregateFunction(Enum):
    """Supported aggregate functions for measures."""
    SUM = "SUM"
    AVG = "AVG"
    COUNT = "COUNT"
    MIN = "MIN"
    MAX = "MAX"


@dataclass
class Level:
    """
    A level in a dimension hierarchy.
    
    Attributes:
        name: Level name (e.g., 'day', 'month', 'year')
        column: Database column name
        order: Position in hierarchy (0 = finest, higher = coarser)
    """
    name: str
    column: str
    order: int
    
    def __hash__(self):
        return hash((self.name, self.column, self.order))
    
    def __eq__(self, other):
        if not isinstance(other, Level):
            return False
        return self.name == other.name and self.column == other.column


@dataclass
class Dimension:
    """
    A dimension with hierarchy of levels.
    
    Following Definition 1: Each dimension D_i has levels <L_i^0, ..., L_i^{h_i}>
    where L_i^0 is finest granularity and L_i^{h_i} is coarsest (often 'All').
    
    Attributes:
        name: Dimension name (e.g., 'Time', 'Product', 'Region')
        levels: Ordered list of levels from finest to coarsest
        table: Dimension table name in database
        key_column: Foreign key column in fact table
    """
    name: str
    levels: List[Level]
    table: str
    key_column: str
    
    def __post_init__(self):
        # Ensure levels are ordered by their order attribute
        self.levels = sorted(self.levels, key=lambda l: l.order)
    
    @property
    def finest_level(self) -> Level:
        """Return the finest granularity level (L_i^0)."""
        return self.levels[0]
    
    @property
    def coarsest_level(self) -> Level:
        """Return the coarsest granularity level (L_i^{h_i})."""
        return self.levels[-1]
    
    def get_level_by_name(self, name: str) -> Optional[Level]:
        """Get level by name."""
        for level in self.levels:
            if level.name == name:
                return level
        return None
    
    def get_parent_level(self, level: Level) -> Optional[Level]:
        """Get the parent (coarser) level for roll-up."""
        idx = self.levels.index(level)
        if idx < len(self.levels) - 1:
            return self.levels[idx + 1]
        return None
    
    def get_child_level(self, level: Level) -> Optional[Level]:
        """Get the child (finer) level for drill-down."""
        idx = self.levels.index(level)
        if idx > 0:
            return self.levels[idx - 1]
        return None
    
    def can_drill_down(self, level: Level) -> bool:
        """Check if drill-down is possible from this level."""
        return self.get_child_level(level) is not None
    
    def can_roll_up(self, level: Level) -> bool:
        """Check if roll-up is possible from this level."""
        return self.get_parent_level(level) is not None


@dataclass
class Measure:
    """
    A numerical measure in the cube.
    
    Attributes:
        name: Measure name (e.g., 'revenue', 'units_sold')
        column: Database column name in fact table
        default_agg: Default aggregation function
    """
    name: str
    column: str
    default_agg: AggregateFunction = AggregateFunction.SUM
    
    def __hash__(self):
        return hash(self.name)


@dataclass
class CubeSchema:
    """
    Cube Schema S = <D, M> as defined in Definition 1.
    
    Attributes:
        name: Cube name (e.g., 'SalesCube')
        dimensions: Set of dimensions D = {D_1, ..., D_n}
        measures: Set of measures M = {m_1, ..., m_k}
        fact_table: Name of the fact table
    """
    name: str
    dimensions: List[Dimension]
    measures: List[Measure]
    fact_table: str
    
    def get_dimension(self, name: str) -> Optional[Dimension]:
        """Get dimension by name."""
        for dim in self.dimensions:
            if dim.name == name:
                return dim
        return None
    
    def get_measure(self, name: str) -> Optional[Measure]:
        """Get measure by name."""
        for m in self.measures:
            if m.name == name:
                return m
        return None
    
    @property
    def dimension_names(self) -> List[str]:
        """Return list of dimension names."""
        return [d.name for d in self.dimensions]
    
    @property
    def measure_names(self) -> List[str]:
        """Return list of measure names."""
        return [m.name for m in self.measures]
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize schema to dictionary."""
        return {
            "name": self.name,
            "fact_table": self.fact_table,
            "dimensions": [
                {
                    "name": d.name,
                    "table": d.table,
                    "key_column": d.key_column,
                    "levels": [
                        {"name": l.name, "column": l.column, "order": l.order}
                        for l in d.levels
                    ]
                }
                for d in self.dimensions
            ],
            "measures": [
                {"name": m.name, "column": m.column, "default_agg": m.default_agg.value}
                for m in self.measures
            ]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CubeSchema":
        """Deserialize schema from dictionary."""
        dimensions = [
            Dimension(
                name=d["name"],
                table=d["table"],
                key_column=d["key_column"],
                levels=[Level(l["name"], l["column"], l["order"]) for l in d["levels"]]
            )
            for d in data["dimensions"]
        ]
        measures = [
            Measure(m["name"], m["column"], AggregateFunction(m["default_agg"]))
            for m in data["measures"]
        ]
        return cls(
            name=data["name"],
            dimensions=dimensions,
            measures=measures,
            fact_table=data["fact_table"]
        )
