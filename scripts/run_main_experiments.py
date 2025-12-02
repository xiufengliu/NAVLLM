#!/usr/bin/env python3
"""
Run REAL experiments with DeepSeek API.
Supports resuming from partial results.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Set
import pandas as pd
import numpy as np
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from navllm.cube.schema import CubeSchema, Dimension, Measure, Level, AggregateFunction
from navllm.cube.view import CubeView, Filter
from navllm.cube.engine import PandasCubeEngine
from navllm.cube.actions import DrillDownAction, RollUpAction
from navllm.llm.client import DeepSeekClient
from navllm.nav.session import Session
from navllm.nav.recommender import NavLLMRecommender, RecommenderConfig

# API Key
API_KEY = "sk-0f250dd04a4b446cb08b17a40a11519d"

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).parent.parent / "results"
CONFIG_DIR = Path(__file__).parent.parent / "configs"

# Output file - single file for all results
OUTPUT_FILE = RESULTS_DIR / "real_experiment_results.csv"


@dataclass
class Task:
    task_id: str
    cube_name: str
    description: str
    target_measure: str
    analysis_type: str
    target_views: List[Dict]


def load_tasks() -> List[Task]:
    """Load tasks from JSON."""
    task_file = CONFIG_DIR / "expanded_tasks.json"
    with open(task_file, 'r') as f:
        data = json.load(f)
    
    tasks = []
    for t in data['tasks']:
        tasks.append(Task(
            task_id=t['task_id'],
            cube_name=t['cube_name'],
            description=t['description'],
            target_measure=t['target_measure'],
            analysis_type=t['analysis_type'],
            target_views=t['target_views']
        ))
    return tasks


def load_completed() -> Set[Tuple[str, str, int]]:
    """Load already completed (task_id, method, run) tuples."""
    if not OUTPUT_FILE.exists():
        return set()
    
    df = pd.read_csv(OUTPUT_FILE)
    completed = set()
    for _, row in df.iterrows():
        completed.add((row['task_id'], row['method'], int(row['run'])))
    return completed


def save_result(result: Dict):
    """Append single result to CSV."""
    df = pd.DataFrame([result])
    
    if OUTPUT_FILE.exists():
        df.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
    else:
        df.to_csv(OUTPUT_FILE, index=False)


def load_cube(cube_name: str):
    """Load cube data."""
    cube_dir = DATA_DIR / cube_name
    
    if cube_name == "m5":
        schema = CubeSchema(
            name="SalesCube",
            fact_table="fact_sales",
            dimensions=[
                Dimension(name="Time", table="dim_time", key_column="date",
                         levels=[Level("month", "month", 0), Level("quarter", "quarter", 1),
                                Level("year", "year", 2), Level("All", "all_time", 3)]),
                Dimension(name="Product", table="dim_product", key_column="item_id",
                         levels=[Level("item", "item_id", 0), Level("department", "dept_id", 1),
                                Level("category", "cat_id", 2), Level("All", "all_product", 3)]),
                Dimension(name="Store", table="dim_store", key_column="store_id",
                         levels=[Level("store", "store_id", 0), Level("state", "state_id", 1),
                                Level("All", "all_store", 2)]),
            ],
            measures=[Measure("units_sold", "units", AggregateFunction.SUM),
                      Measure("revenue", "revenue", AggregateFunction.SUM)]
        )
        fact_df = pd.read_csv(cube_dir / "fact_sales.csv")
        dim_dfs = {
            "Time": pd.read_csv(cube_dir / "dim_time.csv"),
            "Product": pd.read_csv(cube_dir / "dim_product.csv"),
            "Store": pd.read_csv(cube_dir / "dim_store.csv"),
        }
    elif cube_name == "manufacturing":
        schema = CubeSchema(
            name="ManufacturingCube",
            fact_table="fact_production",
            dimensions=[
                Dimension(name="Time", table="dim_time", key_column="time_id",
                         levels=[Level("shift", "shift", 0), Level("day", "day", 1),
                                Level("week", "week", 2), Level("month", "month", 3),
                                Level("All", "all_time", 4)]),
                Dimension(name="Line", table="dim_line", key_column="line_id",
                         levels=[Level("machine", "machine_id", 0), Level("line", "line_id", 1),
                                Level("plant", "plant_id", 2), Level("All", "all_line", 3)]),
                Dimension(name="Product", table="dim_product", key_column="product_id",
                         levels=[Level("variant", "variant_id", 0), Level("family", "family_id", 1),
                                Level("category", "category_id", 2), Level("All", "all_product", 3)]),
            ],
            measures=[Measure("throughput", "throughput", AggregateFunction.SUM),
                      Measure("defect_count", "defect_count", AggregateFunction.SUM),
                      Measure("defect_rate", "defect_rate", AggregateFunction.AVG)]
        )
        fact_df = pd.read_csv(cube_dir / "fact_production.csv")
        dim_dfs = {
            "Time": pd.read_csv(cube_dir / "dim_time.csv"),
            "Line": pd.read_csv(cube_dir / "dim_line.csv"),
            "Product": pd.read_csv(cube_dir / "dim_product.csv"),
        }
    elif cube_name == "air_quality":
        schema = CubeSchema(
            name="AirQualityCube",
            fact_table="fact_air_quality",
            dimensions=[
                Dimension(name="Time", table="dim_time", key_column="time_id",
                         levels=[Level("day", "day", 0), Level("week", "week", 1),
                                Level("month", "month", 2), Level("season", "season", 3),
                                Level("All", "all_time", 4)]),
                Dimension(name="Location", table="dim_location", key_column="station_id",
                         levels=[Level("station", "station_id", 0), Level("district", "district", 1),
                                Level("city", "city", 2), Level("All", "all_location", 3)]),
                Dimension(name="Pollutant", table="dim_pollutant", key_column="pollutant_id",
                         levels=[Level("pollutant", "pollutant_id", 0), Level("All", "all_pollutant", 1)]),
            ],
            measures=[Measure("concentration", "concentration", AggregateFunction.AVG)]
        )
        fact_df = pd.read_csv(cube_dir / "fact_air_quality.csv", nrows=100000)
        dim_dfs = {
            "Time": pd.read_csv(cube_dir / "dim_time.csv"),
            "Location": pd.read_csv(cube_dir / "dim_location.csv"),
            "Pollutant": pd.read_csv(cube_dir / "dim_pollutant.csv"),
        }
    else:
        raise ValueError(f"Unknown cube: {cube_name}")
    
    return schema, fact_df, dim_dfs


def check_hit(view: CubeView, target_views: List[Dict], strict: bool = True) -> bool:
    """Check if view matches any target."""
    for target in target_views:
        target_levels = target.get('levels', {})
        
        if strict:
            if view.levels == target_levels:
                return True
        else:
            total = len(target_levels)
            matches = sum(1 for dim, level in target_levels.items() 
                         if view.levels.get(dim) == level)
            if total > 0 and matches >= total * 0.66:
                return True
    return False


def compute_redundancy(views: List[CubeView]) -> float:
    """Compute redundancy."""
    if len(views) <= 1:
        return 0.0
    
    duplicates = 0
    for i in range(1, len(views)):
        for j in range(i):
            if views[i].levels == views[j].levels:
                duplicates += 1
                break
    
    return duplicates / (len(views) - 1)


class RandomNavigator:
    """Random baseline."""
    def __init__(self, schema, engine):
        self.schema = schema
        self.engine = engine
    
    def select_next_view(self, session, utterance, target_measure):
        current = session.current_view
        actions = []
        for dim_name, level_name in current.levels.items():
            dim = self.schema.get_dimension(dim_name)
            level = dim.get_level_by_name(level_name)
            if level.order > 0:
                child = dim.levels[level.order - 1]
                actions.append(DrillDownAction(dim_name, child.name))
        
        if not actions:
            return current, None
        
        action = np.random.choice(actions)
        return action.apply(current), action


class HeuristicNavigator:
    """Data-driven baseline (I_data only)."""
    def __init__(self, schema, engine):
        self.schema = schema
        self.engine = engine
    
    def select_next_view(self, session, utterance, target_measure):
        current = session.current_view
        actions = []
        for dim_name, level_name in current.levels.items():
            dim = self.schema.get_dimension(dim_name)
            level = dim.get_level_by_name(level_name)
            if level.order > 0:
                child = dim.levels[level.order - 1]
                actions.append(DrillDownAction(dim_name, child.name))
        
        if not actions:
            return current, None
        
        parent_result = self.engine.materialize_view(current)
        best_score = -1
        best_action = actions[0]
        
        for action in actions:
            candidate = action.apply(current)
            try:
                score = self.engine.compute_data_interestingness(candidate, parent_result, target_measure)
            except:
                score = 0
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action.apply(current), best_action


class NavLLMWrapper:
    """Wrapper for NavLLM recommender."""
    def __init__(self, recommender):
        self.recommender = recommender
    
    def select_next_view(self, session, utterance, target_measure):
        recs = self.recommender.recommend(session, utterance, target_measure)
        if recs:
            return recs[0].view, recs[0].action
        return session.current_view, None


class LLMOnlyNavigator:
    """LLM-only baseline (no data interestingness)."""
    def __init__(self, schema, engine, llm_client):
        self.schema = schema
        self.engine = engine
        config = RecommenderConfig(lambda_data=0.0, lambda_pref=0.8, lambda_div=0.2)
        self.recommender = NavLLMRecommender(schema, engine, llm_client, config)
    
    def select_next_view(self, session, utterance, target_measure):
        recs = self.recommender.recommend(session, utterance, target_measure)
        if recs:
            return recs[0].view, recs[0].action
        return session.current_view, None


def run_session(navigator, task, schema, engine, budget=12):
    """Run single navigation session."""
    session = Session()
    initial = CubeView.create_overview(schema)
    session.add_state(initial, task.description)
    
    hit_strict = False
    hit_relaxed = False
    steps_strict = None
    steps_relaxed = None
    i_data_scores = [0.0]
    views = [initial]
    
    parent_result = engine.materialize_view(initial)
    
    for step in range(budget):
        current = session.current_view
        
        next_view, action = navigator.select_next_view(session, task.description, task.target_measure)
        
        if next_view == current or next_view is None:
            break
        
        # Compute I_data
        try:
            i_data = engine.compute_data_interestingness(next_view, parent_result, task.target_measure)
        except:
            i_data = 0.0
        i_data_scores.append(i_data)
        
        session.add_state(next_view, "")
        views.append(next_view)
        
        # Check hits
        if not hit_strict and check_hit(next_view, task.target_views, strict=True):
            hit_strict = True
            steps_strict = step + 1
        
        if not hit_relaxed and check_hit(next_view, task.target_views, strict=False):
            hit_relaxed = True
            steps_relaxed = step + 1
    
    return {
        'hit_strict': hit_strict,
        'hit_relaxed': hit_relaxed,
        'steps_strict': steps_strict,
        'steps_relaxed': steps_relaxed,
        'cumulative_i_data': sum(i_data_scores),
        'redundancy': compute_redundancy(views),
        'num_steps': len(views) - 1,
    }


def main():
    print("="*70)
    print("RUNNING REAL EXPERIMENTS WITH DEEPSEEK API")
    print("="*70)
    print(f"Started: {datetime.now()}")
    print(f"Output: {OUTPUT_FILE}")
    print()
    
    # Load tasks
    tasks = load_tasks()
    print(f"Loaded {len(tasks)} tasks")
    
    # Load completed experiments
    completed = load_completed()
    print(f"Already completed: {len(completed)} sessions")
    
    # Config
    num_runs = 5
    budget = 12
    methods = ['NavLLM', 'LLM-Only', 'Heuristic-Nav', 'Random-Nav']
    
    # LLM client (shared)
    llm_client = DeepSeekClient(api_key=API_KEY)
    
    # Cache for loaded cubes
    cube_cache = {}
    
    total = len(tasks) * len(methods) * num_runs
    done = len(completed)
    
    print(f"Total sessions: {total}, Remaining: {total - done}")
    print()
    
    for task in tasks:
        # Load cube (with caching)
        if task.cube_name not in cube_cache:
            print(f"Loading cube: {task.cube_name}...")
            try:
                cube_cache[task.cube_name] = load_cube(task.cube_name)
            except Exception as e:
                print(f"  ERROR loading cube {task.cube_name}: {e}")
                continue
        
        schema, fact_df, dim_dfs = cube_cache[task.cube_name]
        engine = PandasCubeEngine(schema, fact_df, dim_dfs)
        
        for method in methods:
            for run in range(num_runs):
                # Skip if already done
                if (task.task_id, method, run) in completed:
                    continue
                
                done += 1
                print(f"[{done}/{total}] {task.task_id} | {method} | run {run+1}...", end=' ', flush=True)
                
                try:
                    # Create navigator
                    if method == 'NavLLM':
                        config = RecommenderConfig(
                            lambda_data=0.4, lambda_pref=0.4, lambda_div=0.2, top_k=3
                        )
                        recommender = NavLLMRecommender(schema, engine, llm_client, config)
                        navigator = NavLLMWrapper(recommender)
                    elif method == 'LLM-Only':
                        navigator = LLMOnlyNavigator(schema, engine, llm_client)
                    elif method == 'Heuristic-Nav':
                        navigator = HeuristicNavigator(schema, engine)
                    elif method == 'Random-Nav':
                        navigator = RandomNavigator(schema, engine)
                    
                    # Run session
                    start = time.time()
                    result = run_session(navigator, task, schema, engine, budget)
                    elapsed = time.time() - start
                    
                    # Add metadata
                    result['task_id'] = task.task_id
                    result['cube_name'] = task.cube_name
                    result['method'] = method
                    result['run'] = run
                    result['elapsed_sec'] = elapsed
                    result['timestamp'] = datetime.now().isoformat()
                    result['error'] = None
                    
                    # Save immediately
                    save_result(result)
                    
                    print(f"Hit={result['hit_strict']}, I_data={result['cumulative_i_data']:.2f}, {elapsed:.1f}s")
                    
                    # Rate limiting for LLM methods
                    if method in ['NavLLM', 'LLM-Only']:
                        time.sleep(0.3)
                        
                except Exception as e:
                    print(f"ERROR: {e}")
                    traceback.print_exc()
                    
                    # Save error result
                    error_result = {
                        'task_id': task.task_id,
                        'cube_name': task.cube_name,
                        'method': method,
                        'run': run,
                        'hit_strict': False,
                        'hit_relaxed': False,
                        'steps_strict': None,
                        'steps_relaxed': None,
                        'cumulative_i_data': 0.0,
                        'redundancy': 0.0,
                        'num_steps': 0,
                        'elapsed_sec': 0,
                        'timestamp': datetime.now().isoformat(),
                        'error': str(e)
                    }
                    save_result(error_result)
    
    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    if OUTPUT_FILE.exists():
        df = pd.read_csv(OUTPUT_FILE)
        df_valid = df[df['error'].isna()]
        
        for method in methods:
            method_df = df_valid[df_valid['method'] == method]
            if len(method_df) > 0:
                print(f"\n{method} (n={len(method_df)}):")
                print(f"  Hit Rate (strict):  {method_df['hit_strict'].mean():.3f}")
                print(f"  Hit Rate (relaxed): {method_df['hit_relaxed'].mean():.3f}")
                print(f"  Cumulative I_data:  {method_df['cumulative_i_data'].mean():.2f} ± {method_df['cumulative_i_data'].std():.2f}")
                print(f"  Redundancy:         {method_df['redundancy'].mean():.3f}")
        
        # Check for errors
        errors = df[df['error'].notna()]
        if len(errors) > 0:
            print(f"\n⚠ {len(errors)} sessions had errors")
    
    print(f"\n✓ Completed: {datetime.now()}")


if __name__ == "__main__":
    main()
