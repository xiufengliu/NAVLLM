#!/usr/bin/env python3
"""
Example: Running a NavLLM navigation session.

This script demonstrates how to:
1. Set up a cube schema and engine
2. Initialize the NavLLM recommender
3. Run an interactive navigation session
4. Evaluate the session
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np

from navllm.cube.schema import CubeSchema, Dimension, Measure, Level, AggregateFunction
from navllm.cube.view import CubeView
from navllm.cube.engine import PandasCubeEngine
from navllm.llm.client import MockLLMClient
from navllm.nav.session import Session
from navllm.nav.recommender import NavLLMRecommender, RecommenderConfig
from navllm.eval.metrics import MetricsCalculator


def create_sample_data():
    """Create sample sales data for demonstration."""
    np.random.seed(42)
    
    # Simple denormalized data
    stores = ['CA_1', 'CA_2', 'TX_1', 'TX_2', 'WI_1']
    states = {'CA_1': 'CA', 'CA_2': 'CA', 'TX_1': 'TX', 'TX_2': 'TX', 'WI_1': 'WI'}
    categories = ['Foods', 'Household', 'Hobbies']
    
    records = []
    for year in [2022, 2023]:
        for month in range(1, 13):
            for store in stores:
                for cat in categories:
                    base = 1000 + np.random.normal(0, 200)
                    
                    # Seasonal effect
                    if month in [11, 12]:
                        base *= 1.3
                    elif month in [1, 2]:
                        base *= 0.8
                    
                    # Store effect
                    if store.startswith('CA'):
                        base *= 1.2
                    
                    # Category effect
                    if cat == 'Foods':
                        base *= 1.5
                    
                    # Anomaly
                    if year == 2023 and month == 6 and store == 'TX_1' and cat == 'Foods':
                        base *= 0.3
                    
                    units = max(0, int(base))
                    revenue = units * np.random.uniform(8, 12)
                    
                    records.append({
                        'date_id': f"{year}-{month:02d}",
                        'store_id': store,
                        'item_id': f"{cat}_item",
                        'units': units,
                        'revenue': revenue,
                        # Denormalized dimension attributes
                        'month': month,
                        'year': year,
                        'all_time': 'All',
                        'state_id': states[store],
                        'all_store': 'All',
                        'cat_id': cat,
                        'all_product': 'All',
                    })
    
    fact_df = pd.DataFrame(records)
    
    # Create minimal dimension tables (for the engine's merge)
    dim_time = fact_df[['date_id', 'month', 'year', 'all_time']].drop_duplicates()
    dim_store = fact_df[['store_id', 'state_id', 'all_store']].drop_duplicates()
    dim_product = fact_df[['item_id', 'cat_id', 'all_product']].drop_duplicates()
    
    return fact_df, {
        'Time': dim_time,
        'Store': dim_store,
        'Product': dim_product
    }


def create_schema():
    """Create a simplified SalesCube schema."""
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
        key_column="item_id",
        levels=[
            Level("item", "item_id", 0),
            Level("category", "cat_id", 1),
            Level("All", "all_product", 2),
        ]
    )
    
    store_dim = Dimension(
        name="Store",
        table="dim_store",
        key_column="store_id",
        levels=[
            Level("store", "store_id", 0),
            Level("state", "state_id", 1),
            Level("All", "all_store", 2),
        ]
    )
    
    return CubeSchema(
        name="SalesCube",
        fact_table="fact_sales",
        dimensions=[time_dim, product_dim, store_dim],
        measures=[
            Measure("units_sold", "units", AggregateFunction.SUM),
            Measure("revenue", "revenue", AggregateFunction.SUM),
        ]
    )


def run_demo():
    """Run a demonstration navigation session."""
    print("=" * 60)
    print("NavLLM Demo: Conversational Cube Navigation")
    print("=" * 60)
    
    # Setup
    print("\n1. Setting up cube and data...")
    schema = create_schema()
    fact_df, dim_dfs = create_sample_data()
    
    engine = PandasCubeEngine(schema, fact_df, dim_dfs)
    
    # Use mock LLM for demo
    llm_client = MockLLMClient(default_scores=[0.8, 0.6, 0.4])
    
    config = RecommenderConfig(
        lambda_data=0.4,
        lambda_pref=0.4,
        lambda_div=0.2,
        top_k=3,
        generate_explanations=False
    )
    
    recommender = NavLLMRecommender(schema, engine, llm_client, config)
    
    # Create session
    print("\n2. Starting navigation session...")
    session = Session()
    
    # Initial view: overview
    initial_view = CubeView.create_overview(schema)
    session.add_state(initial_view, "Show me the overall sales data")
    
    result = engine.materialize_view(initial_view)
    print(f"\nInitial view: {initial_view.describe()}")
    print(f"Result: {result.row_count} rows")
    print(result.data.head())
    
    # Simulate navigation steps
    utterances = [
        "I want to see where sales dropped",
        "Show me the breakdown by state",
        "Focus on Texas stores",
    ]
    
    target_measure = "revenue"
    
    for i, utterance in enumerate(utterances, 1):
        print(f"\n{'='*60}")
        print(f"Step {i}: User says: \"{utterance}\"")
        print("="*60)
        
        # Get recommendations
        recommendations = recommender.recommend(session, utterance, target_measure)
        
        print(f"\nTop-{len(recommendations)} Recommendations:")
        for j, rec in enumerate(recommendations, 1):
            print(f"\n  {j}. {rec.action.describe()}")
            print(f"     Utility: {rec.utility:.3f} (I_data={rec.i_data:.3f}, I_pref={rec.i_pref:.3f}, I_div={rec.i_div:.3f})")
            print(f"     {rec.summary.description}")
        
        # Select top recommendation
        if recommendations:
            selected = recommendations[0]
            session.record_action(selected.action)
            session.record_recommendations(
                [r.to_dict() for r in recommendations],
                selected_idx=0
            )
            
            # Move to new view
            new_view = selected.view
            session.add_state(new_view, utterance, selected.summary)
            
            result = engine.materialize_view(new_view)
            print(f"\n  → Selected: {selected.action.describe()}")
            print(f"     New view has {result.row_count} rows")
            if result.row_count <= 10:
                print(result.data)
    
    # Evaluate session
    print("\n" + "="*60)
    print("Session Evaluation")
    print("="*60)
    
    calculator = MetricsCalculator()
    
    # For demo, create a simple target view
    target_view = CubeView(
        schema=schema,
        levels={"Time": "year", "Product": "category", "Store": "state"},
        filters=[]
    )
    
    # Compute I_data for each visited view
    i_data_scores = []
    visited = session.get_visited_views()
    for i, view in enumerate(visited):
        if i == 0:
            i_data_scores.append(0)
        else:
            parent_result = engine.materialize_view(visited[i-1])
            i_data = engine.compute_data_interestingness(view, parent_result, target_measure)
            i_data_scores.append(i_data)
    
    eval_result = calculator.evaluate_session(
        session, [target_view], i_data_scores, task_id="demo_task"
    )
    
    print(f"\nSession ID: {session.session_id}")
    print(f"Total steps: {eval_result.total_steps}")
    print(f"Hit target: {eval_result.hit}")
    print(f"Cumulative I_data: {eval_result.cumulative_i_data:.3f}")
    print(f"Redundancy: {eval_result.redundancy:.2%}")
    print(f"Coverage: {eval_result.coverage} clusters")
    
    # Save session
    session.metadata["task"] = "demo"
    os.makedirs("logs", exist_ok=True)
    session.save("logs/demo_session.json")
    print(f"\nSession saved to logs/demo_session.json")
    
    print("\n" + "="*60)
    print("Demo complete!")
    print("="*60)


if __name__ == "__main__":
    run_demo()
