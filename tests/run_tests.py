#!/usr/bin/env python3
"""
Simple test runner without pytest dependency.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np

from navllm.cube.schema import CubeSchema, Dimension, Measure, Level, AggregateFunction
from navllm.cube.view import CubeView, Filter
from navllm.cube.actions import DrillDownAction, RollUpAction, SliceAction, generate_candidate_actions
from navllm.cube.engine import PandasCubeEngine


def create_test_schema():
    """Create a simple test schema."""
    time_dim = Dimension(
        name="Time",
        table="dim_time",
        key_column="date_id",
        levels=[
            Level("month", "month", 0),
            Level("year", "year", 1),
            Level("All", "all_time", 2),
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


def create_test_data(schema):
    """Create sample data for testing."""
    fact_df = pd.DataFrame({
        'date_id': [1, 1, 2, 2, 3, 3],
        'product_id': ['A', 'B', 'A', 'B', 'A', 'B'],
        'sales_amount': [100, 200, 150, 250, 120, 180],
        'month': [1, 1, 1, 1, 2, 2],
        'year': [2023, 2023, 2023, 2023, 2023, 2023],
        'all_time': ['All'] * 6,
        'category': ['Cat1', 'Cat2', 'Cat1', 'Cat2', 'Cat1', 'Cat2'],
        'all_product': ['All'] * 6,
    })
    
    dim_time = pd.DataFrame({
        'date_id': [1, 2, 3],
        'month': [1, 1, 2],
        'year': [2023, 2023, 2023],
        'all_time': ['All', 'All', 'All'],
    })
    
    dim_product = pd.DataFrame({
        'product_id': ['A', 'B'],
        'category': ['Cat1', 'Cat2'],
        'all_product': ['All', 'All'],
    })
    
    return fact_df, {'Time': dim_time, 'Product': dim_product}


def test_schema_creation():
    """Test schema creation."""
    schema = create_test_schema()
    assert schema.name == "TestCube"
    assert len(schema.dimensions) == 2
    assert len(schema.measures) == 1
    print("✓ test_schema_creation passed")


def test_dimension_levels():
    """Test dimension level navigation."""
    schema = create_test_schema()
    time_dim = schema.get_dimension("Time")
    
    assert time_dim.finest_level.name == "month"
    assert time_dim.coarsest_level.name == "All"
    
    month_level = time_dim.get_level_by_name("month")
    assert time_dim.can_roll_up(month_level)
    assert not time_dim.can_drill_down(month_level)
    print("✓ test_dimension_levels passed")


def test_view_creation():
    """Test view creation."""
    schema = create_test_schema()
    view = CubeView(
        schema=schema,
        levels={"Time": "month", "Product": "category"}
    )
    assert view.levels["Time"] == "month"
    assert view.levels["Product"] == "category"
    print("✓ test_view_creation passed")


def test_overview_view():
    """Test overview view creation."""
    schema = create_test_schema()
    view = CubeView.create_overview(schema)
    assert view.levels["Time"] == "All"
    assert view.levels["Product"] == "All"
    print("✓ test_overview_view passed")


def test_drill_down():
    """Test drill-down action."""
    schema = create_test_schema()
    view = CubeView(
        schema=schema,
        levels={"Time": "year", "Product": "All"}
    )
    
    action = DrillDownAction(dimension="Time")
    assert action.is_applicable(view)
    
    new_view = action.apply(view)
    assert new_view.levels["Time"] == "month"
    assert new_view.levels["Product"] == "All"
    print("✓ test_drill_down passed")


def test_roll_up():
    """Test roll-up action."""
    schema = create_test_schema()
    view = CubeView(
        schema=schema,
        levels={"Time": "month", "Product": "product"}
    )
    
    action = RollUpAction(dimension="Time")
    assert action.is_applicable(view)
    
    new_view = action.apply(view)
    assert new_view.levels["Time"] == "year"
    print("✓ test_roll_up passed")


def test_slice():
    """Test slice action."""
    schema = create_test_schema()
    view = CubeView(
        schema=schema,
        levels={"Time": "year", "Product": "category"}
    )
    
    action = SliceAction(dimension="Time", value=2023)
    assert action.is_applicable(view)
    
    new_view = action.apply(view)
    assert len(new_view.filters) == 1
    assert new_view.filters[0].value == 2023
    print("✓ test_slice passed")


def test_generate_candidates():
    """Test candidate action generation."""
    schema = create_test_schema()
    view = CubeView(
        schema=schema,
        levels={"Time": "year", "Product": "category"}
    )
    
    actions = generate_candidate_actions(view)
    action_types = [a.action_type.value for a in actions]
    
    assert "drill_down" in action_types
    assert "roll_up" in action_types
    print("✓ test_generate_candidates passed")


def test_materialize_view():
    """Test view materialization."""
    schema = create_test_schema()
    fact_df, dim_dfs = create_test_data(schema)
    engine = PandasCubeEngine(schema, fact_df, dim_dfs)
    
    view = CubeView(
        schema=schema,
        levels={"Time": "month", "Product": "All"}
    )
    
    result = engine.materialize_view(view)
    assert result.row_count > 0
    assert "sales" in result.data.columns
    print("✓ test_materialize_view passed")


def test_compute_interestingness():
    """Test data interestingness computation."""
    schema = create_test_schema()
    fact_df, dim_dfs = create_test_data(schema)
    engine = PandasCubeEngine(schema, fact_df, dim_dfs)
    
    parent_view = CubeView(
        schema=schema,
        levels={"Time": "All", "Product": "All"}
    )
    child_view = CubeView(
        schema=schema,
        levels={"Time": "month", "Product": "All"}
    )
    
    parent_result = engine.materialize_view(parent_view)
    i_data = engine.compute_data_interestingness(child_view, parent_result, "sales")
    
    assert isinstance(i_data, float)
    assert i_data >= 0
    print("✓ test_compute_interestingness passed")


def run_all_tests():
    """Run all tests."""
    print("Running NavLLM tests...\n")
    
    tests = [
        test_schema_creation,
        test_dimension_levels,
        test_view_creation,
        test_overview_view,
        test_drill_down,
        test_roll_up,
        test_slice,
        test_generate_candidates,
        test_materialize_view,
        test_compute_interestingness,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} ERROR: {e}")
            failed += 1
    
    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*40}")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
