"""
Cube Engine: Executes OLAP queries and computes statistics.

This module handles all numerical computations including:
- View materialization (Res(V, C))
- Data-driven interestingness I_data(V | m*)
- Statistics for candidate views
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

from navllm.cube.schema import CubeSchema, AggregateFunction
from navllm.cube.view import CubeView, Filter


@dataclass
class ViewResult:
    """
    Result of materializing a cube view: Res(V, C).
    
    Attributes:
        view: The view specification
        data: DataFrame containing the aggregated results
        row_count: Number of tuples in result
        statistics: Computed statistics for each measure
    """
    view: CubeView
    data: pd.DataFrame
    row_count: int
    statistics: Dict[str, Dict[str, float]]  # measure -> {mean, std, min, max, ...}
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for LLM consumption."""
        return {
            "row_count": self.row_count,
            "measures": self.statistics,
            "grouping": self.view.levels,
            "filters": [f.to_dict() for f in self.view.filters]
        }


class CubeEngine(ABC):
    """
    Abstract base class for cube engines.
    
    Implementations can use different backends (PostgreSQL, DuckDB, pandas, etc.)
    """
    
    def __init__(self, schema: CubeSchema):
        self.schema = schema
        self._cache: Dict[str, ViewResult] = {}
    
    @abstractmethod
    def materialize_view(self, view: CubeView) -> ViewResult:
        """Execute the view query and return results."""
        pass
    
    @abstractmethod
    def get_dimension_members(self, dimension: str, level: str, 
                              filters: List[Filter] = None) -> List[Any]:
        """Get distinct member values for a dimension level."""
        pass
    
    def compute_data_interestingness(self, view: CubeView, 
                                      parent_result: ViewResult,
                                      target_measure: str) -> float:
        """
        Compute I_data(V | m*) as defined in Definition 4.
        
        Uses deviation-based measure:
        I_data = (1/|T(V)|) * Σ |v_m(τ) - v_m(par(τ))| / (v_m(par(τ)) + ε)
        
        Args:
            view: The candidate view V
            parent_result: Result of parent view par(V)
            target_measure: The target measure m*
        
        Returns:
            Data-driven interestingness score
        """
        result = self.materialize_view(view)
        
        if result.row_count == 0:
            return 0.0
        
        # Get parent statistics
        parent_mean = parent_result.statistics.get(target_measure, {}).get("mean", 0)
        epsilon = 1e-6
        
        # Compute deviation for each tuple
        measure_values = result.data[target_measure].values
        deviations = np.abs(measure_values - parent_mean) / (np.abs(parent_mean) + epsilon)
        
        return float(np.mean(deviations))
    
    def compute_view_statistics(self, result: ViewResult, 
                                 target_measure: str) -> Dict[str, Any]:
        """
        Compute statistics for a view result.
        
        Returns statistics suitable for LLM consumption.
        """
        if result.row_count == 0:
            return {"empty": True}
        
        data = result.data[target_measure]
        stats = result.statistics.get(target_measure, {})
        
        # Compute outliers (> 2σ from mean)
        mean = stats.get("mean", data.mean())
        std = stats.get("std", data.std())
        outlier_count = int(np.sum(np.abs(data - mean) > 2 * std)) if std > 0 else 0
        
        # Coefficient of variation
        cv = std / (abs(mean) + 1e-6) if mean != 0 else 0
        
        return {
            "row_count": result.row_count,
            "mean": float(mean),
            "std": float(std),
            "min": float(data.min()),
            "max": float(data.max()),
            "cv": float(cv),
            "outlier_count": outlier_count,
            "deviation_category": self._categorize_deviation(cv)
        }
    
    def _categorize_deviation(self, cv: float) -> str:
        """Categorize coefficient of variation into bins."""
        if cv < 0.1:
            return "low"
        elif cv < 0.3:
            return "medium"
        else:
            return "high"
    
    def clear_cache(self):
        """Clear the view result cache."""
        self._cache.clear()


class PandasCubeEngine(CubeEngine):
    """
    In-memory cube engine using pandas DataFrames.
    
    Suitable for small-to-medium datasets and prototyping.
    """
    
    def __init__(self, schema: CubeSchema, 
                 fact_df: pd.DataFrame,
                 dimension_dfs: Dict[str, pd.DataFrame]):
        """
        Initialize with data.
        
        Args:
            schema: Cube schema
            fact_df: Fact table as DataFrame
            dimension_dfs: Dict mapping dimension name -> dimension DataFrame
        """
        super().__init__(schema)
        self.fact_df = fact_df
        self.dimension_dfs = dimension_dfs
        self._merged_df = self._build_merged_df()
    
    def _build_merged_df(self) -> pd.DataFrame:
        """Build denormalized DataFrame by joining fact with dimensions."""
        df = self.fact_df.copy()
        
        for dim in self.schema.dimensions:
            dim_df = self.dimension_dfs[dim.name]
            # Get columns to add (exclude key column and columns already in df)
            existing_cols = set(df.columns)
            new_cols = [c for c in dim_df.columns if c not in existing_cols or c == dim.key_column]
            if len(new_cols) > 1:  # More than just the key column
                dim_df_subset = dim_df[new_cols].drop_duplicates()
                df = df.merge(dim_df_subset, on=dim.key_column, how="left", suffixes=('', '_dim'))
        
        return df
    
    def materialize_view(self, view: CubeView) -> ViewResult:
        """Execute view query using pandas."""
        # Check cache
        cache_key = view.view_id
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        df = self._merged_df.copy()
        
        # Apply filters
        for f in view.filters:
            dim = self.schema.get_dimension(f.dimension)
            level = dim.get_level_by_name(f.level)
            col = level.column
            
            if f.operator == "=":
                df = df[df[col] == f.value]
            elif f.operator == "IN":
                df = df[df[col].isin(f.value)]
            elif f.operator == "BETWEEN":
                df = df[(df[col] >= f.value[0]) & (df[col] <= f.value[1])]
            elif f.operator == "<":
                df = df[df[col] < f.value]
            elif f.operator == ">":
                df = df[df[col] > f.value]
        
        # Build grouping columns
        group_cols = []
        for dim_name, level_name in view.levels.items():
            if level_name != "All":
                dim = self.schema.get_dimension(dim_name)
                level = dim.get_level_by_name(level_name)
                group_cols.append(level.column)
        
        # Aggregate
        if group_cols:
            agg_dict = {}
            for measure_name, agg_func in view.aggregations.items():
                measure = self.schema.get_measure(measure_name)
                # Map SQL aggregate names to pandas
                agg_name = agg_func.value.lower()
                if agg_name == "avg":
                    agg_name = "mean"
                agg_dict[measure.column] = agg_name
            
            result_df = df.groupby(group_cols, as_index=False).agg(agg_dict)
            # Rename columns to measure names
            rename_map = {
                self.schema.get_measure(m).column: m 
                for m in view.aggregations.keys()
            }
            result_df = result_df.rename(columns=rename_map)
        else:
            # No grouping - aggregate to single row
            result_dict = {}
            for measure_name, agg_func in view.aggregations.items():
                measure = self.schema.get_measure(measure_name)
                col = measure.column
                if agg_func == AggregateFunction.SUM:
                    result_dict[measure_name] = [df[col].sum()]
                elif agg_func == AggregateFunction.AVG:
                    result_dict[measure_name] = [df[col].mean()]
                elif agg_func == AggregateFunction.COUNT:
                    result_dict[measure_name] = [df[col].count()]
                elif agg_func == AggregateFunction.MIN:
                    result_dict[measure_name] = [df[col].min()]
                elif agg_func == AggregateFunction.MAX:
                    result_dict[measure_name] = [df[col].max()]
            result_df = pd.DataFrame(result_dict)
        
        # Compute statistics
        statistics = {}
        for measure_name in view.aggregations.keys():
            if measure_name in result_df.columns:
                col_data = result_df[measure_name]
                statistics[measure_name] = {
                    "mean": float(col_data.mean()) if len(col_data) > 0 else 0,
                    "std": float(col_data.std()) if len(col_data) > 1 else 0,
                    "min": float(col_data.min()) if len(col_data) > 0 else 0,
                    "max": float(col_data.max()) if len(col_data) > 0 else 0,
                }
        
        result = ViewResult(
            view=view,
            data=result_df,
            row_count=len(result_df),
            statistics=statistics
        )
        
        # Cache result
        self._cache[cache_key] = result
        return result
    
    def get_dimension_members(self, dimension: str, level: str,
                              filters: List[Filter] = None) -> List[Any]:
        """Get distinct member values for a dimension level."""
        dim = self.schema.get_dimension(dimension)
        level_obj = dim.get_level_by_name(level)
        
        df = self._merged_df.copy()
        
        # Apply filters if provided
        if filters:
            for f in filters:
                dim_f = self.schema.get_dimension(f.dimension)
                level_f = dim_f.get_level_by_name(f.level)
                col = level_f.column
                if f.operator == "=":
                    df = df[df[col] == f.value]
        
        return df[level_obj.column].dropna().unique().tolist()
