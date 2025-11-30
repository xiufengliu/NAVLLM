"""
Unit tests for the cube module.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from navllm.cube.schema import CubeSchema, Dimension, Measure, Level, AggregateFunction
from navllm.cube.view import CubeView, Filter
from navllm.cube.actions import DrillDownAction, RollUpAction, SliceAction, generate_candidate_actions
from navllm.cube.engine import PandasCubeEngine


@pytest.fixture
def simple_schema():
    """Create a simple test schema."""
    time_dim = Dimension(
        name="Time",
        table="dim_time",
        key_column="date_id",
        levels=[
            Level("day", "day", 0),
            Level("month", "month", 1),
            Level("year", "year", 2),
            Level("All", "all_time", 3),
        ]
    )
    
    product_dim = Dimension(
        name="Product",
        table="dim_product",
        key_column="product_id",
        levels=[
            Level("product", "product_id", 0),
            Level("category", "category", 1),
            Level("All", "all_product", 2),
        ]
    )
    
    return CubeSchema(
        name="TestCube",
        fact_table="fact_sales",
        dimensions=[time_dim, product_dim],
        measures=[
            Measure("sales", "sales_amount", AggregateFunction.SUM),
        ]
    )


@pytest.fixture
def sample_data(simple_schema):
    """Create sample data for testing."""
    fact_df = pd.DataFrame({
        'date_id': [1, 1, 2, 2, 3, 3],
        'product_id': ['A', 'B', 'A', 'B', 'A', 'B'],
        'sales_amount': [100, 200, 150, 250, 120, 180],
    })
    
    dim_time = pd.DataFrame({
        'date_id': [1, 2, 3],
        'day': ['2023-01-01', '2023-01-02', '2023-02-01'],
        'month': ['2023-01', '2023-01', '2023-02'],
        'year': [2023, 2023, 2023],
        'all_time': ['All', 'All', 'All'],
    })
    
    dim_product = pd.DataFrame({
        'product_id': ['A', 'B'],
        'category': ['Cat1', 'Cat2'],
        'all_product': ['All', 'All'],
    })
    
    return fact_df, {'Time': dim_time, 'Product': dim_product}


class TestCubeSchema:
    def test_schema_creation(self, simple_schema):
        assert simple_schema.name == "TestCube"
        assert len(simple_schema.dimensions) == 2
        assert len(simple_schema.measures) == 1
    
    def test_get_dimension(self, simple_schema):
        time_dim = simple_schema.get_dimension("Time")
        assert time_dim is not None
        assert time_dim.name == "Time"
        
        assert simple_schema.get_dimension("NonExistent") is None
    
    def test_dimension_levels(self, simple_schema):
        time_dim = simple_schema.get_dimension("Time")
        assert time_dim.finest_level.name == "day"
        assert time_dim.coarsest_level.name == "All"
        
        day_level = time_dim.get_level_by_name("day")
        assert time_dim.can_roll_up(day_level)
        assert not time_dim.can_drill_down(day_level)


class TestCubeView:
    def test_view_creation(self, simple_schema):
        view = CubeView(
            schema=simple_schema,
            levels={"Time": "month", "Product": "category"}
        )
        assert view.levels["Time"] == "month"
        assert view.levels["Product"] == "category"
    
    def test_overview_view(self, simple_schema):
        view = CubeView.create_overview(simple_schema)
        assert view.levels["Time"] == "All"
        assert view.levels["Product"] == "All"
    
    def test_view_describe(self, simple_schema):
        view = CubeView(
            schema=simple_schema,
            levels={"Time": "month", "Product": "category"}
        )
        desc = view.describe()
        assert "Time.month" in desc
        assert "Product.category" in desc
    
    def test_view_with_filter(self, simple_schema):
        view = CubeView(
            schema=simple_schema,
            levels={"Time": "month", "Product": "category"},
            filters=[Filter("Time", "year", "=", 2023)]
        )
        assert len(view.filters) == 1
        assert view.filters[0].value == 2023


class TestNavigationActions:
    def test_drill_down(self, simple_schema):
        view = CubeView(
            schema=simple_schema,
            levels={"Time": "year", "Product": "All"}
        )
        
        action = DrillDownAction(dimension="Time")
        assert action.is_applicable(view)
        
        new_view = action.apply(view)
        assert new_view.levels["Time"] == "month"
        assert new_view.levels["Product"] == "All"
    
    def test_roll_up(self, simple_schema):
        view = CubeView(
            schema=simple_schema,
            levels={"Time": "day", "Product": "product"}
        )
        
        action = RollUpAction(dimension="Time")
        assert action.is_applicable(view)
        
        new_view = action.apply(view)
        assert new_view.levels["Time"] == "month"
    
    def test_slice(self, simple_schema):
        view = CubeView(
            schema=simple_schema,
            levels={"Time": "year", "Product": "category"}
        )
        
        action = SliceAction(dimension="Time", value=2023)
        assert action.is_applicable(view)
        
        new_view = action.apply(view)
        assert len(new_view.filters) == 1
        assert new_view.filters[0].value == 2023
    
    def test_generate_candidates(self, simple_schema):
        view = CubeView(
            schema=simple_schema,
            levels={"Time": "month", "Product": "category"}
        )
        
        actions = generate_candidate_actions(view)
        
        # Should have drill-down and roll-up for both dimensions
        action_types = [a.action_type.value for a in actions]
        assert "drill_down" in action_types
        assert "roll_up" in action_types


class TestPandasCubeEngine:
    def test_materialize_view(self, simple_schema, sample_data):
        fact_df, dim_dfs = sample_data
        engine = PandasCubeEngine(simple_schema, fact_df, dim_dfs)
        
        view = CubeView(
            schema=simple_schema,
            levels={"Time": "month", "Product": "All"}
        )
        
        result = engine.materialize_view(view)
        assert result.row_count > 0
        assert "sales" in result.data.columns
    
    def test_compute_interestingness(self, simple_schema, sample_data):
        fact_df, dim_dfs = sample_data
        engine = PandasCubeEngine(simple_schema, fact_df, dim_dfs)
        
        parent_view = CubeView(
            schema=simple_schema,
            levels={"Time": "All", "Product": "All"}
        )
        child_view = CubeView(
            schema=simple_schema,
            levels={"Time": "month", "Product": "All"}
        )
        
        parent_result = engine.materialize_view(parent_view)
        i_data = engine.compute_data_interestingness(child_view, parent_result, "sales")
        
        assert isinstance(i_data, float)
        assert i_data >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
