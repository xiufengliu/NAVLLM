"""
Cube View definitions following Definition 3 in Section 3.2.

A cube view V = <L_vec, φ, A> specifies:
- L_vec: chosen level for each dimension
- φ: selection predicate (filters)
- A: aggregation function for each measure
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from copy import deepcopy
import hashlib
import json

from navllm.cube.schema import (
    CubeSchema, Dimension, Level, Measure, AggregateFunction
)


@dataclass
class Filter:
    """A selection predicate on a dimension."""
    dimension: str
    level: str
    operator: str  # '=', 'IN', 'BETWEEN', '<', '>', etc.
    value: Any
    
    def to_sql(self, schema: CubeSchema) -> str:
        """Convert filter to SQL WHERE clause fragment."""
        dim = schema.get_dimension(self.dimension)
        level = dim.get_level_by_name(self.level)
        col = f"{dim.table}.{level.column}"
        
        if self.operator == "=":
            if isinstance(self.value, str):
                return f"{col} = '{self.value}'"
            return f"{col} = {self.value}"
        elif self.operator == "IN":
            vals = ", ".join(f"'{v}'" if isinstance(v, str) else str(v) for v in self.value)
            return f"{col} IN ({vals})"
        elif self.operator == "BETWEEN":
            return f"{col} BETWEEN {self.value[0]} AND {self.value[1]}"
        else:
            return f"{col} {self.operator} {self.value}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension,
            "level": self.level,
            "operator": self.operator,
            "value": self.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Filter":
        return cls(**data)


@dataclass
class CubeView:
    """
    A cube view V = <L_vec, φ, A> as defined in Definition 3.
    
    Attributes:
        schema: Reference to the cube schema
        levels: Dict mapping dimension name -> chosen level name
        filters: List of selection predicates (φ)
        aggregations: Dict mapping measure name -> aggregate function
    """
    schema: CubeSchema
    levels: Dict[str, str]  # dimension_name -> level_name
    filters: List[Filter] = field(default_factory=list)
    aggregations: Dict[str, AggregateFunction] = field(default_factory=dict)
    
    def __post_init__(self):
        # Initialize default aggregations if not provided
        if not self.aggregations:
            self.aggregations = {
                m.name: m.default_agg for m in self.schema.measures
            }
    
    @property
    def view_id(self) -> str:
        """Generate unique identifier for this view."""
        content = json.dumps({
            "levels": self.levels,
            "filters": [f.to_dict() for f in self.filters],
            "aggregations": {k: v.value for k, v in self.aggregations.items()}
        }, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def get_level(self, dimension_name: str) -> Level:
        """Get the Level object for a dimension."""
        dim = self.schema.get_dimension(dimension_name)
        return dim.get_level_by_name(self.levels[dimension_name])
    
    def get_grouping_columns(self) -> List[Tuple[str, str]]:
        """Return list of (table.column) for GROUP BY."""
        columns = []
        for dim_name, level_name in self.levels.items():
            dim = self.schema.get_dimension(dim_name)
            level = dim.get_level_by_name(level_name)
            if level.name != "All":  # Skip 'All' level
                columns.append((dim.table, level.column))
        return columns
    
    def to_sql(self) -> str:
        """
        Generate SQL query for this view.
        Returns the SQL string to materialize Res(V, C).
        """
        # SELECT clause: grouping columns + aggregated measures
        select_parts = []
        group_cols = self.get_grouping_columns()
        
        for table, col in group_cols:
            select_parts.append(f"{table}.{col}")
        
        for measure_name, agg_func in self.aggregations.items():
            measure = self.schema.get_measure(measure_name)
            select_parts.append(
                f"{agg_func.value}({self.schema.fact_table}.{measure.column}) AS {measure_name}"
            )
        
        # FROM clause: fact table with dimension joins
        from_parts = [self.schema.fact_table]
        join_parts = []
        
        for dim in self.schema.dimensions:
            join_parts.append(
                f"JOIN {dim.table} ON {self.schema.fact_table}.{dim.key_column} = {dim.table}.{dim.key_column}"
            )
        
        # WHERE clause: filters
        where_parts = [f.to_sql(self.schema) for f in self.filters]
        
        # GROUP BY clause
        group_by_parts = [f"{t}.{c}" for t, c in group_cols]
        
        # Build query
        sql = f"SELECT {', '.join(select_parts)}\n"
        sql += f"FROM {from_parts[0]}\n"
        sql += "\n".join(join_parts) + "\n"
        
        if where_parts:
            sql += f"WHERE {' AND '.join(where_parts)}\n"
        
        if group_by_parts:
            sql += f"GROUP BY {', '.join(group_by_parts)}"
        
        return sql
    
    def copy(self) -> "CubeView":
        """Create a deep copy of this view."""
        return CubeView(
            schema=self.schema,
            levels=dict(self.levels),
            filters=[Filter(**f.to_dict()) for f in self.filters],
            aggregations=dict(self.aggregations)
        )
    
    def describe(self) -> str:
        """Generate human-readable description of the view."""
        parts = []
        
        # Grouping
        group_desc = []
        for dim_name, level_name in self.levels.items():
            if level_name != "All":
                group_desc.append(f"{dim_name}.{level_name}")
        if group_desc:
            parts.append(f"Grouped by: {', '.join(group_desc)}")
        else:
            parts.append("Aggregated to total (no grouping)")
        
        # Filters
        if self.filters:
            filter_desc = []
            for f in self.filters:
                filter_desc.append(f"{f.dimension}.{f.level} {f.operator} {f.value}")
            parts.append(f"Filtered: {', '.join(filter_desc)}")
        
        # Measures
        measure_desc = [f"{agg.value}({m})" for m, agg in self.aggregations.items()]
        parts.append(f"Measures: {', '.join(measure_desc)}")
        
        return "; ".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize view to dictionary."""
        return {
            "levels": self.levels,
            "filters": [f.to_dict() for f in self.filters],
            "aggregations": {k: v.value for k, v in self.aggregations.items()}
        }
    
    @classmethod
    def from_dict(cls, schema: CubeSchema, data: Dict[str, Any]) -> "CubeView":
        """Deserialize view from dictionary."""
        return cls(
            schema=schema,
            levels=data["levels"],
            filters=[Filter.from_dict(f) for f in data.get("filters", [])],
            aggregations={
                k: AggregateFunction(v) for k, v in data.get("aggregations", {}).items()
            }
        )
    
    @classmethod
    def create_overview(cls, schema: CubeSchema) -> "CubeView":
        """Create initial overview view at coarsest level for all dimensions."""
        levels = {
            dim.name: dim.coarsest_level.name
            for dim in schema.dimensions
        }
        return cls(schema=schema, levels=levels)
    
    def __hash__(self):
        return hash(self.view_id)
    
    def __eq__(self, other):
        if not isinstance(other, CubeView):
            return False
        return self.view_id == other.view_id
