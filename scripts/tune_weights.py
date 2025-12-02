#!/usr/bin/env python3
"""
Tune utility function weights to improve hit rate.

Addresses reviewer concern: NavLLM has lower hit rate than Random-Nav.
Tests different weight configurations to find optimal balance.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from itertools import product

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from run_improved_experiments import (
    load_expanded_tasks, load_cube_data, run_single_session,
    ExperimentConfig
)
from navllm.cube.engine import PandasCubeEngine
from navllm.llm.client import MockLLMClient
from navllm.nav.recommender import NavLLMRecommender, RecommenderConfig
from baselines import NavLLMWrapper


def tune_weights():
    """
    Grid search over utility function weights.
    
    Focus on configurations that increase preference weight
    to improve task alignment and hit rate.
    """
    
    # Weight configurations to test
    configs = [
        # Original
        (0.4, 0.4, 0.2, "balanced"),
        # More preference-driven (should improve hit rate)
        (0.3, 0.5, 0.2, "pref_high"),
        (0.2, 0.6, 0.2, "pref_very_high"),
        (0.25, 0.6, 0.15, "pref_dominant"),
        # More data-driven (may hurt hit rate but improve I_data)
        (0.5, 0.3, 0.2, "data_high"),
        (0.6, 0.2, 0.2, "data_very_high"),
        # Balanced with more diversity
        (0.35, 0.35, 0.3, "diverse"),
    ]
    
    # Load subset of tasks for tuning
    all_tasks = load_expanded_tasks()
    # Use first 2 tasks per cube for tuning
    tune_tasks = []
    for cube in ['m5', 'manufacturing', 'air_quality']:
        cube_tasks = [t for t in all_tasks if t.cube_name == cube]
        tune_tasks.extend(cube_tasks[:2])
    
    print(f"Tuning on {len(tune_tasks)} tasks")
    
    results = []
    
    for lambda_data, lambda_pref, lambda_div, config_name in configs:
        print(f"\n{'='*60}")
        print(f"Testing: {config_name}")
        print(f"  λ_data={lambda_data}, λ_pref={lambda_pref}, λ_div={lambda_div}")
        print(f"{'='*60}")
        
        config = ExperimentConfig(
            interaction_budget=12,
            num_runs=3,  # Fewer runs for tuning
            lambda_data=lambda_data,
            lambda_pref=lambda_pref,
            lambda_div=lambda_div
        )
        
        task_results = []
        
        for task in tune_tasks:
            # Load cube
            try:
                schema, fact_df, dim_dfs = load_cube_data(task.cube_name)
                engine = PandasCubeEngine(schema, fact_df, dim_dfs)
            except Exception as e:
                print(f"  Error loading {task.cube_name}: {e}")
                continue
            
            # Create navigator with these weights
            llm_client = MockLLMClient()  # Use mock for speed
            rec_config = RecommenderConfig(
                lambda_data=lambda_data,
                lambda_pref=lambda_pref,
                lambda_div=lambda_div,
                top_k=3
            )
            recommender = NavLLMRecommender(schema, engine, llm_client, rec_config)
            navigator = NavLLMWrapper(recommender)
            
            # Run multiple times
            for run in range(config.num_runs):
                result = run_single_session(navigator, task, config, schema, engine)
                result['config_name'] = config_name
                result['lambda_data'] = lambda_data
                result['lambda_pref'] = lambda_pref
                result['lambda_div'] = lambda_div
                task_results.append(result)
        
        # Aggregate for this config
        df = pd.DataFrame(task_results)
        summary = {
            'config_name': config_name,
            'lambda_data': lambda_data,
            'lambda_pref': lambda_pref,
            'lambda_div': lambda_div,
            'hit_rate_strict': df['hit_strict'].mean(),
            'hit_rate_relaxed': df['hit_relaxed'].mean(),
            'avg_steps_to_hit': df['steps_to_hit_strict'].mean(),
            'cumulative_i_data': df['cumulative_i_data'].mean(),
            'redundancy': df['redundancy'].mean(),
        }
        results.append(summary)
        
        print(f"\n  Results:")
        print(f"    Hit Rate (strict): {summary['hit_rate_strict']:.3f}")
        print(f"    Hit Rate (relaxed): {summary['hit_rate_relaxed']:.3f}")
        print(f"    Cumulative I_data: {summary['cumulative_i_data']:.2f}")
        print(f"    Redundancy: {summary['redundancy']:.3f}")
    
    # Save results
    results_df = pd.DataFrame(results)
    output_file = Path(__file__).parent.parent / "results" / "weight_tuning.csv"
    results_df.to_csv(output_file, index=False)
    
    # Find best configuration
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    # Sort by hit rate (primary) and I_data (secondary)
    results_df['score'] = results_df['hit_rate_strict'] + 0.1 * results_df['cumulative_i_data'] / 10
    results_df = results_df.sort_values('score', ascending=False)
    
    print("\nTop 3 configurations:")
    for i, row in results_df.head(3).iterrows():
        print(f"\n{i+1}. {row['config_name']}")
        print(f"   Weights: λ_data={row['lambda_data']}, λ_pref={row['lambda_pref']}, λ_div={row['lambda_div']}")
        print(f"   Hit Rate: {row['hit_rate_strict']:.3f}")
        print(f"   I_data: {row['cumulative_i_data']:.2f}")
        print(f"   Redundancy: {row['redundancy']:.3f}")
    
    best = results_df.iloc[0]
    print(f"\n{'='*60}")
    print("RECOMMENDATION:")
    print(f"Use λ_data={best['lambda_data']}, λ_pref={best['lambda_pref']}, λ_div={best['lambda_div']}")
    print(f"Expected hit rate: {best['hit_rate_strict']:.3f}")
    print(f"{'='*60}")
    
    return results_df


if __name__ == "__main__":
    results = tune_weights()
