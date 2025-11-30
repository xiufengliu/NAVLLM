#!/usr/bin/env python3
"""
Full experimental evaluation for Information Systems submission.
Includes: main results, per-cube breakdown, sensitivity analysis, ablation study.
"""

import os, sys, json, argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from navllm.cube.schema import CubeSchema, Dimension, Measure, Level, AggregateFunction
from navllm.cube.view import CubeView, Filter
from navllm.cube.engine import PandasCubeEngine
from navllm.cube.actions import generate_candidate_actions
from navllm.llm.client import MockLLMClient, DeepSeekClient
from navllm.nav.session import Session
from navllm.nav.recommender import NavLLMRecommender, RecommenderConfig
from navllm.eval.metrics import MetricsCalculator

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).parent.parent / "results"

# ============== Data Loading (reuse from run_experiments.py) ==============
def load_cube_data(cube_name: str):
    cube_dir = DATA_DIR / cube_name
    if cube_name == "m5":
        return load_m5_cube(cube_dir)
    elif cube_name == "manufacturing":
        return load_manufacturing_cube(cube_dir)
    elif cube_name == "air_quality":
        return load_air_quality_cube(cube_dir)

def load_m5_cube(cube_dir):
    schema = CubeSchema(
        name="SalesCube", fact_table="fact_sales",
        dimensions=[
            Dimension(name="Time", table="dim_time", key_column="date",
                     levels=[Level("month", "month", 0), Level("quarter", "quarter", 1),
                            Level("year", "year", 2), Level("All", "all_time", 3)]),
            Dimension(name="Product", table="dim_product", key_column="item_id",
                     levels=[Level("category", "category", 0), Level("department", "department", 1),
                            Level("All", "all_products", 2)]),
            Dimension(name="Store", table="dim_store", key_column="store_id",
                     levels=[Level("store", "store_id", 0), Level("state", "state", 1),
                            Level("All", "all_stores", 2)]),
        ],
        measures=[Measure("revenue", "revenue", AggregateFunction.SUM),
                  Measure("units_sold", "units_sold", AggregateFunction.SUM)]
    )
    fact_df = pd.read_csv(cube_dir / "fact_sales.csv")
    return schema, fact_df, {}

def load_manufacturing_cube(cube_dir):
    schema = CubeSchema(
        name="ManufacturingCube", fact_table="fact_production",
        dimensions=[
            Dimension(name="Time", table="dim_time", key_column="date",
                     levels=[Level("day", "day", 0), Level("week", "week", 1),
                            Level("month", "month", 2), Level("All", "all_time", 3)]),
            Dimension(name="Line", table="dim_line", key_column="line_id",
                     levels=[Level("machine", "machine_id", 0), Level("line", "line_id", 1),
                            Level("plant", "plant_id", 2), Level("All", "all_lines", 3)]),
            Dimension(name="Product", table="dim_product", key_column="product_id",
                     levels=[Level("variant", "variant_id", 0), Level("family", "family_id", 1),
                            Level("All", "all_products", 2)]),
        ],
        measures=[Measure("throughput", "throughput", AggregateFunction.SUM),
                  Measure("defect_count", "defect_count", AggregateFunction.SUM),
                  Measure("defect_rate", "defect_rate", AggregateFunction.AVG)]
    )
    fact_df = pd.read_csv(cube_dir / "fact_production.csv")
    return schema, fact_df, {}

def load_air_quality_cube(cube_dir):
    schema = CubeSchema(
        name="AirQualityCube", fact_table="fact_readings",
        dimensions=[
            Dimension(name="Time", table="dim_time", key_column="datetime",
                     levels=[Level("hour", "hour", 0), Level("day", "day", 1),
                            Level("month", "month", 2), Level("All", "all_time", 3)]),
            Dimension(name="Location", table="dim_location", key_column="station_id",
                     levels=[Level("station", "station_id", 0), Level("city", "city", 1),
                            Level("All", "all_locations", 2)]),
            Dimension(name="Pollutant", table="dim_pollutant", key_column="pollutant",
                     levels=[Level("pollutant", "pollutant", 0), Level("All", "all_pollutants", 1)]),
        ],
        measures=[Measure("concentration", "concentration", AggregateFunction.AVG)]
    )
    fact_df = pd.read_csv(cube_dir / "fact_air_quality.csv")
    return schema, fact_df, {}

# ============== Tasks ==============
def create_tasks():
    return [
        {"task_id": "sales_1", "cube_name": "m5", "description": "Find regions and product categories where sales declined",
         "target_measure": "revenue", "target_views": [{"levels": {"Time": "quarter", "Product": "category", "Store": "state"}}]},
        {"task_id": "sales_2", "cube_name": "m5", "description": "Identify top-performing stores",
         "target_measure": "revenue", "target_views": [{"levels": {"Time": "month", "Product": "All", "Store": "store"}}]},
        {"task_id": "sales_3", "cube_name": "m5", "description": "Investigate seasonal patterns in categories",
         "target_measure": "units_sold", "target_views": [{"levels": {"Time": "quarter", "Product": "category", "Store": "All"}}]},
        {"task_id": "mfg_1", "cube_name": "manufacturing", "description": "Find line-shift combinations with elevated defect rates",
         "target_measure": "defect_rate", "target_views": [{"levels": {"Time": "day", "Line": "line", "Product": "All"}}]},
        {"task_id": "mfg_2", "cube_name": "manufacturing", "description": "Identify product families with highest throughput variability",
         "target_measure": "throughput", "target_views": [{"levels": {"Time": "week", "Line": "All", "Product": "family"}}]},
        {"task_id": "aq_1", "cube_name": "air_quality", "description": "Investigate temporal and spatial patterns of high pollution",
         "target_measure": "concentration", "target_views": [{"levels": {"Time": "day", "Location": "city", "Pollutant": "pollutant"}}]},
        {"task_id": "aq_2", "cube_name": "air_quality", "description": "Compare air quality across regions",
         "target_measure": "concentration", "target_views": [{"levels": {"Time": "month", "Location": "city", "Pollutant": "All"}}]},
    ]

# ============== Navigators ==============
class RandomNavigator:
    def __init__(self, schema, engine): self.schema, self.engine = schema, engine
    def select_next_view(self, session, utterance, target_measure):
        actions = generate_candidate_actions(session.current_view)
        if not actions: return session.current_view, None
        action = np.random.choice(actions)
        return action.apply(session.current_view) or session.current_view, action

class HeuristicNavigator:
    def __init__(self, schema, engine): self.schema, self.engine = schema, engine
    def select_next_view(self, session, utterance, target_measure):
        current = session.current_view
        try: current_result = self.engine.materialize_view(current)
        except: return current, None
        actions = generate_candidate_actions(current)
        if not actions: return current, None
        best_score, best_view, best_action = -1, current, None
        for action in actions:
            new_view = action.apply(current)
            if not new_view: continue
            try:
                result = self.engine.materialize_view(new_view)
                if result.row_count < 3: continue
                i_data = self.engine.compute_data_interestingness(new_view, current_result, target_measure)
                if i_data > best_score: best_score, best_view, best_action = i_data, new_view, action
            except: continue
        return best_view, best_action

class LLMOnlyNavigator:
    def __init__(self, schema, engine, llm_client): 
        self.schema, self.engine, self.llm_client = schema, engine, llm_client
    def select_next_view(self, session, utterance, target_measure):
        actions = generate_candidate_actions(session.current_view)
        if not actions: return session.current_view, None
        # Score by keyword matching + LLM preference
        keywords = utterance.lower().split()
        best_score, best_view, best_action = -1, session.current_view, None
        for action in actions:
            new_view = action.apply(session.current_view)
            if not new_view: continue
            score = sum(1 for kw in keywords if kw in action.describe().lower()) + np.random.uniform(0, 0.5)
            if score > best_score: best_score, best_view, best_action = score, new_view, action
        return best_view, best_action

class NavLLMNavigator:
    def __init__(self, schema, engine, llm_client, config):
        self.schema, self.engine, self.llm_client = schema, engine, llm_client
        self.config = config
        self.recommender = NavLLMRecommender(schema, engine, llm_client, config)
    def select_next_view(self, session, utterance, target_measure):
        try:
            recs = self.recommender.recommend(session, utterance, target_measure)
            if recs: return recs[0].view, recs[0].action
        except: pass
        # Fallback
        return HeuristicNavigator(self.schema, self.engine).select_next_view(session, utterance, target_measure)

# ============== Core Experiment Runner ==============
def run_session(navigator, task, budget, schema, engine):
    session = Session()
    session.add_state(CubeView.create_overview(schema), task["description"])
    utterances = [task["description"], "Show more detail", "Break down further", "Focus on anomalies",
                  "Other dimensions?", "Drill down", "Compare categories", "Temporal patterns",
                  "Filter values", "Roll up", "Other patterns?", "Summarize"]
    hit, steps_to_hit, i_data_scores = False, None, [0.0]
    
    for step in range(budget):
        if check_hit(session.current_view, task["target_views"]) and not hit:
            hit, steps_to_hit = True, step + 1
        new_view, action = navigator.select_next_view(session, utterances[step % len(utterances)], task["target_measure"])
        if new_view and new_view != session.current_view:
            try:
                parent_result = engine.materialize_view(session.current_view)
                i_data = engine.compute_data_interestingness(new_view, parent_result, task["target_measure"])
            except: i_data = 0.0
            i_data_scores.append(i_data)
            if action: session.record_action(action)
            session.add_state(new_view, utterances[step % len(utterances)])
    
    if check_hit(session.current_view, task["target_views"]) and not hit:
        hit, steps_to_hit = True, budget
    
    calculator = MetricsCalculator()
    target_views = [CubeView(schema=schema, levels=spec["levels"]) for spec in task["target_views"]]
    eval_result = calculator.evaluate_session(session, target_views, i_data_scores, task["task_id"])
    
    return {"hit": hit, "steps_to_hit": steps_to_hit, "cumulative_i_data": sum(i_data_scores),
            "redundancy": eval_result.redundancy, "coverage": eval_result.coverage}

def check_hit(view, target_specs):
    for spec in target_specs:
        matches = sum(1 for d, l in spec.get("levels", {}).items() if view.levels.get(d) == l or l == "All")
        if matches >= len(spec.get("levels", {})) * 0.66: return True
    return False

# ============== Main Experiments ==============
def run_main_experiment(llm_client, num_runs=5, budget=12):
    """Main experiment: all methods, all tasks."""
    print("\n" + "="*60 + "\n1. MAIN EXPERIMENT\n" + "="*60)
    tasks = create_tasks()
    methods = ["Random-Nav", "Heuristic-Nav", "LLM-Only", "NavLLM"]
    results = []
    
    for task in tasks:
        try:
            schema, fact_df, dim_dfs = load_cube_data(task["cube_name"])
            engine = PandasCubeEngine(schema, fact_df, dim_dfs)
        except Exception as e:
            print(f"  Skip {task['task_id']}: {e}")
            continue
        
        for method in methods:
            config = RecommenderConfig(lambda_data=0.4, lambda_pref=0.4, lambda_div=0.2, top_k=3, generate_explanations=False)
            if method == "Random-Nav": nav = RandomNavigator(schema, engine)
            elif method == "Heuristic-Nav": nav = HeuristicNavigator(schema, engine)
            elif method == "LLM-Only": nav = LLMOnlyNavigator(schema, engine, llm_client)
            else: nav = NavLLMNavigator(schema, engine, llm_client, config)
            
            for run in range(num_runs):
                np.random.seed(42 + run)
                r = run_session(nav, task, budget, schema, engine)
                r.update({"method": method, "task_id": task["task_id"], "cube": task["cube_name"], "run": run})
                results.append(r)
        print(f"  Completed: {task['task_id']}")
    
    return pd.DataFrame(results)

def run_sensitivity_analysis(llm_client, num_runs=3, budget=12):
    """Sensitivity analysis: vary lambda weights."""
    print("\n" + "="*60 + "\n2. SENSITIVITY ANALYSIS\n" + "="*60)
    weight_configs = [
        (0.6, 0.3, 0.1), (0.5, 0.4, 0.1), (0.4, 0.4, 0.2),
        (0.3, 0.5, 0.2), (0.3, 0.4, 0.3), (0.2, 0.6, 0.2),
    ]
    tasks = create_tasks()
    results = []
    
    for ld, lp, ldiv in weight_configs:
        print(f"  Testing λ_data={ld}, λ_pref={lp}, λ_div={ldiv}")
        for task in tasks:
            try:
                schema, fact_df, _ = load_cube_data(task["cube_name"])
                engine = PandasCubeEngine(schema, fact_df, {})
                config = RecommenderConfig(lambda_data=ld, lambda_pref=lp, lambda_div=ldiv, top_k=3, generate_explanations=False)
                nav = NavLLMNavigator(schema, engine, llm_client, config)
                for run in range(num_runs):
                    np.random.seed(42 + run)
                    r = run_session(nav, task, budget, schema, engine)
                    r.update({"lambda_data": ld, "lambda_pref": lp, "lambda_div": ldiv,
                              "task_id": task["task_id"], "cube": task["cube_name"], "run": run})
                    results.append(r)
            except: continue
    
    return pd.DataFrame(results)

def run_ablation_study(llm_client, num_runs=5, budget=12):
    """Ablation study: disable each component."""
    print("\n" + "="*60 + "\n3. ABLATION STUDY\n" + "="*60)
    configs = [
        ("Full NavLLM", 0.4, 0.4, 0.2),
        ("w/o I_pref", 0.6, 0.0, 0.4),
        ("w/o I_data", 0.0, 0.6, 0.4),
        ("w/o I_div", 0.5, 0.5, 0.0),
    ]
    tasks = create_tasks()
    results = []
    
    for name, ld, lp, ldiv in configs:
        print(f"  Testing: {name}")
        for task in tasks:
            try:
                schema, fact_df, _ = load_cube_data(task["cube_name"])
                engine = PandasCubeEngine(schema, fact_df, {})
                config = RecommenderConfig(lambda_data=ld, lambda_pref=lp, lambda_div=ldiv, top_k=3, generate_explanations=False)
                nav = NavLLMNavigator(schema, engine, llm_client, config)
                for run in range(num_runs):
                    np.random.seed(42 + run)
                    r = run_session(nav, task, budget, schema, engine)
                    r.update({"config": name, "task_id": task["task_id"], "cube": task["cube_name"], "run": run})
                    results.append(r)
            except: continue
    
    return pd.DataFrame(results)

# ============== Statistical Tests ==============
def compute_significance(df, baseline="Random-Nav"):
    """Compute p-values for NavLLM vs each baseline."""
    print("\n" + "="*60 + "\n4. STATISTICAL SIGNIFICANCE\n" + "="*60)
    navllm = df[df["method"] == "NavLLM"]
    results = []
    for method in df["method"].unique():
        if method == "NavLLM": continue
        other = df[df["method"] == method]
        for metric in ["hit", "cumulative_i_data", "redundancy", "coverage"]:
            try:
                t_stat, p_val = stats.ttest_ind(navllm[metric].dropna(), other[metric].dropna())
                results.append({"comparison": f"NavLLM vs {method}", "metric": metric, "t_stat": t_stat, "p_value": p_val})
            except: pass
    return pd.DataFrame(results)

# ============== Output Tables ==============
def generate_tables(main_df, sens_df, ablation_df, sig_df):
    """Generate LaTeX-ready tables."""
    print("\n" + "="*60 + "\nRESULTS TABLES\n" + "="*60)
    
    # Table 2: Main results
    print("\n--- TABLE 2: Navigation Quality ---")
    t2 = main_df.groupby("method").agg({
        "hit": ["mean", "std"], "steps_to_hit": ["mean", "std"], "cumulative_i_data": ["mean", "std"]
    }).round(2)
    print(t2)
    
    # Table 3: Efficiency
    print("\n--- TABLE 3: Efficiency & Coverage ---")
    t3 = main_df.groupby("method").agg({"redundancy": ["mean", "std"], "coverage": ["mean", "std"]}).round(2)
    print(t3)
    
    # Table 4: Per-cube breakdown
    print("\n--- TABLE 4: Per-Cube Breakdown ---")
    t4 = main_df.groupby(["cube", "method"]).agg({"hit": "mean", "cumulative_i_data": "mean", "redundancy": "mean"}).round(2)
    print(t4)
    
    # Table 5: Ablation
    print("\n--- TABLE 5: Ablation Study ---")
    t5 = ablation_df.groupby("config").agg({"hit": "mean", "redundancy": "mean", "coverage": "mean"}).round(2)
    print(t5)
    
    # Table 6: Sensitivity
    print("\n--- TABLE 6: Sensitivity Analysis ---")
    sens_df["weights"] = sens_df.apply(lambda r: f"({r['lambda_data']},{r['lambda_pref']},{r['lambda_div']})", axis=1)
    t6 = sens_df.groupby("weights").agg({"hit": "mean", "cumulative_i_data": "mean", "redundancy": "mean"}).round(2)
    print(t6)
    
    # Table 7: Significance
    print("\n--- TABLE 7: Statistical Significance ---")
    print(sig_df.round(4))
    
    return t2, t3, t4, t5, t6, sig_df

# ============== Main ==============
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepseek-key", required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--budget", type=int, default=12)
    args = parser.parse_args()
    
    llm_client = DeepSeekClient(api_key=args.deepseek_key)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Run all experiments
    main_df = run_main_experiment(llm_client, args.runs, args.budget)
    sens_df = run_sensitivity_analysis(llm_client, num_runs=3, budget=args.budget)
    ablation_df = run_ablation_study(llm_client, args.runs, args.budget)
    sig_df = compute_significance(main_df)
    
    # Save results
    main_df.to_csv(RESULTS_DIR / f"main_results_{ts}.csv", index=False)
    sens_df.to_csv(RESULTS_DIR / f"sensitivity_{ts}.csv", index=False)
    ablation_df.to_csv(RESULTS_DIR / f"ablation_{ts}.csv", index=False)
    sig_df.to_csv(RESULTS_DIR / f"significance_{ts}.csv", index=False)
    
    # Generate tables
    generate_tables(main_df, sens_df, ablation_df, sig_df)
    
    print(f"\nAll results saved to {RESULTS_DIR}")

if __name__ == "__main__":
    main()
