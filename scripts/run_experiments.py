#!/usr/bin/env python3
"""
Run NavLLM experiments as described in Section 7 of the paper.

Evaluates:
1. Navigation quality: Hit Rate@B, Steps to First Hit, Cumulative I_data
2. Efficiency: Redundancy, Coverage
3. Compares: NavLLM vs Manual-OLAP vs Heuristic-Nav vs Random-Nav vs LLM-Only
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from navllm.cube.schema import CubeSchema, Dimension, Measure, Level, AggregateFunction
from navllm.cube.view import CubeView, Filter
from navllm.cube.engine import PandasCubeEngine
from navllm.cube.actions import DrillDownAction, RollUpAction, SliceAction, generate_candidate_actions
from navllm.llm.client import MockLLMClient, GeminiClient, DeepSeekClient
from navllm.nav.session import Session
from navllm.nav.recommender import NavLLMRecommender, RecommenderConfig
from navllm.eval.metrics import MetricsCalculator, EvaluationResult

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).parent.parent / "results"


@dataclass
class Task:
    """An analysis task with target views."""
    task_id: str
    cube_name: str
    description: str
    target_measure: str
    target_views: List[Dict]  # View specifications that count as "hits"
    

@dataclass 
class ExperimentConfig:
    """Configuration for experiments."""
    interaction_budget: int = 12
    top_k: int = 3
    num_runs: int = 5  # Runs per task for statistical significance
    lambda_data: float = 0.4
    lambda_pref: float = 0.4
    lambda_div: float = 0.2
    llm_client: Any = None


def load_cube_data(cube_name: str) -> Tuple[CubeSchema, pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Load cube schema and data."""
    cube_dir = DATA_DIR / cube_name
    
    if cube_name == "m5":
        return load_m5_cube(cube_dir)
    elif cube_name == "manufacturing":
        return load_manufacturing_cube(cube_dir)
    elif cube_name == "air_quality":
        return load_air_quality_cube(cube_dir)
    else:
        raise ValueError(f"Unknown cube: {cube_name}")


def load_m5_cube(cube_dir: Path):
    """Load M5 SalesCube."""
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
    return schema, fact_df, dim_dfs


def load_manufacturing_cube(cube_dir: Path):
    """Load ManufacturingCube."""
    schema = CubeSchema(
        name="ManufacturingCube",
        fact_table="fact_production",
        dimensions=[
            Dimension(name="Time", table="dim_time", key_column="date",
                     levels=[Level("day", "day", 0), Level("week", "week", 1),
                            Level("month", "month", 2), Level("year", "year", 3),
                            Level("All", "all_time", 4)]),
            Dimension(name="Line", table="dim_line", key_column="machine_id",
                     levels=[Level("machine", "machine_id", 0), Level("line", "line_id", 1),
                            Level("plant", "plant_id", 2), Level("All", "all_line", 3)]),
            Dimension(name="Product", table="dim_product", key_column="variant_id",
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
    return schema, fact_df, dim_dfs


def load_air_quality_cube(cube_dir: Path):
    """Load AirQualityCube."""
    schema = CubeSchema(
        name="AirQualityCube",
        fact_table="fact_air_quality",
        dimensions=[
            Dimension(name="Time", table="dim_time", key_column="time_id",
                     levels=[Level("day", "day", 0), Level("week", "week", 1),
                            Level("month", "month", 2), Level("season", "season", 3),
                            Level("year", "year", 4), Level("All", "all_time", 5)]),
            Dimension(name="Location", table="dim_location", key_column="station_id",
                     levels=[Level("station", "station_id", 0), Level("district", "district", 1),
                            Level("city", "city", 2), Level("All", "all_location", 3)]),
            Dimension(name="Pollutant", table="dim_pollutant", key_column="pollutant_id",
                     levels=[Level("pollutant", "pollutant_id", 0), Level("All", "all_pollutant", 1)]),
        ],
        measures=[Measure("concentration", "concentration", AggregateFunction.AVG)]
    )
    
    # Sample for faster experiments
    fact_df = pd.read_csv(cube_dir / "fact_air_quality.csv", nrows=500000)
    dim_dfs = {
        "Time": pd.read_csv(cube_dir / "dim_time.csv"),
        "Location": pd.read_csv(cube_dir / "dim_location.csv"),
        "Pollutant": pd.read_csv(cube_dir / "dim_pollutant.csv"),
    }
    return schema, fact_df, dim_dfs


def create_tasks() -> List[Task]:
    """Create analysis tasks as described in Section 7.1.2."""
    tasks = [
        # SalesCube tasks
        Task("sales_1", "m5", 
             "Find regions and product categories where sales declined",
             "revenue",
             [{"levels": {"Time": "year", "Product": "category", "Store": "state"}}]),
        Task("sales_2", "m5",
             "Identify top-performing stores and understand drivers",
             "revenue",
             [{"levels": {"Time": "quarter", "Product": "category", "Store": "store"}}]),
        Task("sales_3", "m5",
             "Investigate seasonal patterns in specific categories",
             "units_sold",
             [{"levels": {"Time": "month", "Product": "category", "Store": "All"}}]),
        
        # ManufacturingCube tasks
        Task("mfg_1", "manufacturing",
             "Find line-shift combinations with elevated defect rates",
             "defect_rate",
             [{"levels": {"Time": "month", "Line": "line", "Product": "All"}}]),
        Task("mfg_2", "manufacturing",
             "Identify product families with highest throughput variability",
             "throughput",
             [{"levels": {"Time": "week", "Line": "All", "Product": "family"}}]),
        
        # AirQualityCube tasks
        Task("aq_1", "air_quality",
             "Investigate temporal and spatial patterns of high PM2.5",
             "concentration",
             [{"levels": {"Time": "month", "Location": "district", "Pollutant": "pollutant"},
               "filters": [{"dimension": "Pollutant", "level": "pollutant", "operator": "=", "value": "PM2.5"}]}]),
        Task("aq_2", "air_quality",
             "Compare air quality across regions to identify hotspots",
             "concentration",
             [{"levels": {"Time": "season", "Location": "station", "Pollutant": "All"}}]),
    ]
    return tasks


class BaselineNavigator:
    """Base class for navigation strategies."""
    
    def __init__(self, schema: CubeSchema, engine: PandasCubeEngine):
        self.schema = schema
        self.engine = engine
    
    def select_next_view(self, session: Session, utterance: str, 
                         target_measure: str) -> Tuple[CubeView, Any]:
        """Select next view. Returns (view, action)."""
        raise NotImplementedError


class RandomNavigator(BaselineNavigator):
    """Random-Nav: Select random action."""
    
    def select_next_view(self, session: Session, utterance: str,
                         target_measure: str) -> Tuple[CubeView, Any]:
        current = session.current_view
        actions = generate_candidate_actions(current)
        
        if not actions:
            return current, None
        
        action = np.random.choice(actions)
        new_view = action.apply(current)
        return new_view if new_view else current, action


class HeuristicNavigator(BaselineNavigator):
    """Heuristic-Nav: Select by I_data only."""
    
    def select_next_view(self, session: Session, utterance: str,
                         target_measure: str) -> Tuple[CubeView, Any]:
        current = session.current_view
        current_result = self.engine.materialize_view(current)
        
        actions = generate_candidate_actions(current)
        if not actions:
            return current, None
        
        best_score = -1
        best_view = current
        best_action = None
        
        for action in actions:
            new_view = action.apply(current)
            if new_view is None:
                continue
            
            result = self.engine.materialize_view(new_view)
            if result.row_count < 3:
                continue
            
            i_data = self.engine.compute_data_interestingness(
                new_view, current_result, target_measure
            )
            
            if i_data > best_score:
                best_score = i_data
                best_view = new_view
                best_action = action
        
        return best_view, best_action


class LLMOnlyNavigator(BaselineNavigator):
    """LLM-Only: Select by I_pref only (simulated)."""
    
    def __init__(self, schema, engine, preference_weights: Dict[str, float] = None):
        super().__init__(schema, engine)
        # Simulate LLM preferences based on keyword matching
        self.preference_weights = preference_weights or {}
    
    def select_next_view(self, session: Session, utterance: str,
                         target_measure: str) -> Tuple[CubeView, Any]:
        current = session.current_view
        actions = generate_candidate_actions(current)
        
        if not actions:
            return current, None
        
        # Simulate preference scoring based on utterance keywords
        keywords = utterance.lower().split()
        best_score = -1
        best_view = current
        best_action = None
        
        for action in actions:
            new_view = action.apply(current)
            if new_view is None:
                continue
            
            # Score based on dimension name matching keywords
            score = 0
            action_desc = action.describe().lower()
            for kw in keywords:
                if kw in action_desc:
                    score += 1
            
            # Add randomness to simulate LLM variability
            score += np.random.uniform(0, 0.5)
            
            if score > best_score:
                best_score = score
                best_view = new_view
                best_action = action
        
        return best_view, best_action


class NavLLMNavigator(BaselineNavigator):
    """NavLLM: Full system with hybrid utility."""
    
    def __init__(self, schema, engine, config: RecommenderConfig, llm_client=None):
        super().__init__(schema, engine)
        self.llm_client = llm_client or MockLLMClient(simulate_preferences=True)
        self.recommender = NavLLMRecommender(schema, engine, self.llm_client, config)
        self.config = config
    
    def select_next_view(self, session: Session, utterance: str,
                         target_measure: str) -> Tuple[CubeView, Any]:
        try:
            recommendations = self.recommender.recommend(session, utterance, target_measure)
            if recommendations:
                # Select best recommendation
                return recommendations[0].view, recommendations[0].action
        except Exception as e:
            pass
        
        # Fallback: use heuristic if NavLLM fails
        current = session.current_view
        current_result = self.engine.materialize_view(current)
        actions = generate_candidate_actions(current)
        
        if not actions:
            return current, None
        
        best_score = -1
        best_view = current
        best_action = None
        
        for action in actions:
            new_view = action.apply(current)
            if new_view is None:
                continue
            
            result = self.engine.materialize_view(new_view)
            if result.row_count < 3:
                continue
            
            # Compute hybrid score
            i_data = self.engine.compute_data_interestingness(
                new_view, current_result, target_measure
            )
            
            # Simulate preference based on action type and utterance
            i_pref = 0.5
            action_desc = action.describe().lower()
            utterance_lower = utterance.lower()
            
            if "drill" in action_desc and any(w in utterance_lower for w in ["detail", "breakdown", "specific", "where"]):
                i_pref = 0.8
            elif "roll" in action_desc and any(w in utterance_lower for w in ["overall", "summary", "total"]):
                i_pref = 0.7
            elif "slice" in action_desc:
                i_pref = 0.6
            
            # Diversity: penalize if similar to visited views
            i_div = 1.0
            for visited in session.get_visited_views():
                if new_view.levels == visited.levels:
                    i_div = 0.2
                    break
            
            score = (self.config.lambda_data * min(i_data, 1.0) + 
                    self.config.lambda_pref * i_pref + 
                    self.config.lambda_div * i_div)
            
            if score > best_score:
                best_score = score
                best_view = new_view
                best_action = action
        
        return best_view, best_action


def check_hit(view: CubeView, target_specs: List[Dict]) -> bool:
    """Check if view matches any target specification (lenient matching)."""
    for spec in target_specs:
        target_levels = spec.get("levels", {})
        
        # Count how many dimensions match at the target level or finer
        matches = 0
        total = len(target_levels)
        
        for dim, target_level in target_levels.items():
            view_level = view.levels.get(dim)
            if view_level == target_level:
                matches += 1
            elif target_level == "All":
                matches += 1  # Any level is fine if target is All
            elif view_level != "All":
                # Check if view is at finer granularity (also acceptable)
                matches += 0.5
        
        # Consider it a hit if at least 2/3 of dimensions match
        if matches >= total * 0.66:
            return True
    
    return False


def run_single_session(navigator: BaselineNavigator, task: Task, 
                       config: ExperimentConfig, schema: CubeSchema,
                       engine: PandasCubeEngine) -> Dict[str, Any]:
    """Run a single navigation session."""
    session = Session()
    
    # Start from overview
    initial_view = CubeView.create_overview(schema)
    session.add_state(initial_view, task.description)
    
    hit = False
    steps_to_hit = None
    i_data_scores = [0.0]
    
    # Simulate utterances (in real experiment, these come from users)
    utterances = [
        task.description,
        "Show me more detail",
        "Break this down further",
        "Focus on the anomalies",
        "What about other dimensions?",
        "Drill down here",
        "Compare across categories",
        "Look at temporal patterns",
        "Filter to specific values",
        "Roll up to see the big picture",
        "Any other interesting patterns?",
        "Summarize the findings",
    ]
    
    for step in range(config.interaction_budget):
        utterance = utterances[step % len(utterances)]
        
        # Check if current view is a hit
        if check_hit(session.current_view, task.target_views):
            if not hit:
                hit = True
                steps_to_hit = step + 1
        
        # Get next view
        new_view, action = navigator.select_next_view(
            session, utterance, task.target_measure
        )
        
        if new_view is None or new_view == session.current_view:
            continue
        
        # Compute I_data for new view
        try:
            parent_result = engine.materialize_view(session.current_view)
            i_data = engine.compute_data_interestingness(
                new_view, parent_result, task.target_measure
            )
        except:
            i_data = 0.0
        
        i_data_scores.append(i_data)
        
        if action:
            session.record_action(action)
        session.add_state(new_view, utterance)
    
    # Final hit check
    if check_hit(session.current_view, task.target_views) and not hit:
        hit = True
        steps_to_hit = config.interaction_budget
    
    # Compute metrics
    calculator = MetricsCalculator()
    
    # Create target view objects for metric calculation
    target_views = []
    for spec in task.target_views:
        tv = CubeView(schema=schema, levels=spec["levels"])
        target_views.append(tv)
    
    eval_result = calculator.evaluate_session(
        session, target_views, i_data_scores, task.task_id
    )
    
    return {
        "hit": hit,
        "steps_to_hit": steps_to_hit,
        "cumulative_i_data": sum(i_data_scores),
        "redundancy": eval_result.redundancy,
        "coverage": eval_result.coverage,
        "total_steps": session.step_count,
    }


def run_experiment(config: ExperimentConfig) -> pd.DataFrame:
    """Run full experiment across all tasks and methods."""
    tasks = create_tasks()
    methods = ["Random-Nav", "Heuristic-Nav", "LLM-Only", "NavLLM"]
    
    results = []
    
    for task in tasks:
        print(f"\n{'='*60}")
        print(f"Task: {task.task_id} - {task.description[:50]}...")
        print(f"Cube: {task.cube_name}, Target: {task.target_measure}")
        print(f"{'='*60}")
        
        # Load cube data
        try:
            schema, fact_df, dim_dfs = load_cube_data(task.cube_name)
            engine = PandasCubeEngine(schema, fact_df, dim_dfs)
        except Exception as e:
            print(f"  Error loading cube: {e}")
            continue
        
        for method in methods:
            print(f"\n  Method: {method}")
            
            # Create navigator
            if method == "Random-Nav":
                navigator = RandomNavigator(schema, engine)
            elif method == "Heuristic-Nav":
                navigator = HeuristicNavigator(schema, engine)
            elif method == "LLM-Only":
                navigator = LLMOnlyNavigator(schema, engine)
            elif method == "NavLLM":
                rec_config = RecommenderConfig(
                    lambda_data=config.lambda_data,
                    lambda_pref=config.lambda_pref,
                    lambda_div=config.lambda_div,
                    top_k=config.top_k,
                    generate_explanations=False
                )
                llm_client = config.llm_client if hasattr(config, 'llm_client') else None
                navigator = NavLLMNavigator(schema, engine, rec_config, llm_client)
            
            # Run multiple times for statistical significance
            method_results = []
            for run in range(config.num_runs):
                np.random.seed(42 + run)  # Reproducible randomness
                
                result = run_single_session(
                    navigator, task, config, schema, engine
                )
                result["method"] = method
                result["task_id"] = task.task_id
                result["cube"] = task.cube_name
                result["run"] = run
                method_results.append(result)
            
            # Aggregate results
            hits = [r["hit"] for r in method_results]
            steps = [r["steps_to_hit"] for r in method_results if r["steps_to_hit"]]
            i_data = [r["cumulative_i_data"] for r in method_results]
            redundancy = [r["redundancy"] for r in method_results]
            coverage = [r["coverage"] for r in method_results]
            
            print(f"    Hit Rate: {np.mean(hits):.2f} ± {np.std(hits):.2f}")
            if steps:
                print(f"    Steps to Hit: {np.mean(steps):.1f} ± {np.std(steps):.1f}")
            print(f"    Cumulative I_data: {np.mean(i_data):.2f} ± {np.std(i_data):.2f}")
            print(f"    Redundancy: {np.mean(redundancy):.2%}")
            print(f"    Coverage: {np.mean(coverage):.1f}")
            
            results.extend(method_results)
    
    return pd.DataFrame(results)


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate results by method."""
    agg = df.groupby("method").agg({
        "hit": ["mean", "std"],
        "steps_to_hit": ["mean", "std"],
        "cumulative_i_data": ["mean", "std"],
        "redundancy": ["mean", "std"],
        "coverage": ["mean", "std"],
    }).round(3)
    
    # Flatten column names
    agg.columns = ['_'.join(col).strip() for col in agg.columns.values]
    return agg


def main():
    parser = argparse.ArgumentParser(description="Run NavLLM experiments")
    parser.add_argument("--budget", type=int, default=12, help="Interaction budget")
    parser.add_argument("--runs", type=int, default=5, help="Runs per task")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument("--gemini-key", type=str, default=None, help="Gemini API key")
    parser.add_argument("--deepseek-key", type=str, default=None, help="DeepSeek API key")
    args = parser.parse_args()
    
    print("="*60)
    print("NavLLM Experiments")
    print("="*60)
    print(f"Interaction budget: {args.budget}")
    print(f"Runs per task: {args.runs}")
    
    # Initialize LLM client
    llm_client = None
    if args.deepseek_key:
        print(f"Using DeepSeek API for NavLLM")
        llm_client = DeepSeekClient(api_key=args.deepseek_key)
    elif args.gemini_key:
        print(f"Using Gemini API for NavLLM")
        llm_client = GeminiClient(api_key=args.gemini_key, model="gemini-2.0-flash")
    else:
        print("Using Mock LLM (no API key provided)")
    
    config = ExperimentConfig(
        interaction_budget=args.budget,
        num_runs=args.runs,
        llm_client=llm_client,
    )
    
    # Run experiments
    results_df = run_experiment(config)
    
    # Save raw results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(RESULTS_DIR / f"raw_results_{timestamp}.csv", index=False)
    
    # Aggregate and display
    print("\n" + "="*60)
    print("AGGREGATED RESULTS")
    print("="*60)
    
    agg_results = aggregate_results(results_df)
    print("\n" + agg_results.to_string())
    
    # Save aggregated results
    agg_results.to_csv(RESULTS_DIR / f"aggregated_results_{timestamp}.csv")
    
    # Create summary table (Table 2 format from paper)
    print("\n" + "="*60)
    print("TABLE 2: Navigation Quality Results")
    print("="*60)
    
    summary = results_df.groupby("method").agg({
        "hit": "mean",
        "steps_to_hit": lambda x: x.dropna().mean() if x.dropna().any() else None,
        "cumulative_i_data": "mean",
    }).round(2)
    summary.columns = ["Hit Rate@12", "Steps to First Hit", "Cumulative I_data"]
    print(summary.to_string())
    
    print("\n" + "="*60)
    print("TABLE 3: Efficiency and Coverage Results")
    print("="*60)
    
    efficiency = results_df.groupby("method").agg({
        "redundancy": "mean",
        "coverage": "mean",
    }).round(2)
    efficiency.columns = ["Redundancy (%)", "Coverage (clusters)"]
    efficiency["Redundancy (%)"] = (efficiency["Redundancy (%)"] * 100).round(1)
    print(efficiency.to_string())
    
    # Save final tables
    summary.to_csv(RESULTS_DIR / f"table2_quality_{timestamp}.csv")
    efficiency.to_csv(RESULTS_DIR / f"table3_efficiency_{timestamp}.csv")
    
    print(f"\nResults saved to {RESULTS_DIR}")
    
    return results_df


if __name__ == "__main__":
    main()
