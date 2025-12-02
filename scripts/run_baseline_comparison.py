#!/usr/bin/env python3
"""
Re-run main experiments with properly differentiated baselines.
20 tasks × 2 runs = 40 sessions per method.
"""
import os, sys, json, time
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from navllm.cube.schema import CubeSchema, Dimension, Measure, Level, AggregateFunction
from navllm.cube.view import CubeView
from navllm.cube.engine import PandasCubeEngine
from navllm.llm.client import DeepSeekClient
from navllm.nav.session import Session
from navllm.nav.recommender import NavLLMRecommender, RecommenderConfig
from baseline_methods import HeuristicNav, SeeDBNav, GreedyBestNav, BFSNav, RandomNav

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).parent.parent / "results"
CONFIG_DIR = Path(__file__).parent.parent / "configs"

def load_tasks():
    with open(CONFIG_DIR / "expanded_tasks.json") as f:
        return json.load(f)['tasks']

def load_cube(cube_name):
    cube_dir = DATA_DIR / cube_name
    if cube_name == "m5":
        schema = CubeSchema(
            name="SalesCube", fact_table="fact_sales",
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
        dim_dfs = {"Time": pd.read_csv(cube_dir / "dim_time.csv"),
                   "Product": pd.read_csv(cube_dir / "dim_product.csv"),
                   "Store": pd.read_csv(cube_dir / "dim_store.csv")}
    elif cube_name == "manufacturing":
        schema = CubeSchema(
            name="ManufacturingCube", fact_table="fact_production",
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
        dim_dfs = {"Time": pd.read_csv(cube_dir / "dim_time.csv"),
                   "Line": pd.read_csv(cube_dir / "dim_line.csv"),
                   "Product": pd.read_csv(cube_dir / "dim_product.csv")}
    elif cube_name == "air_quality":
        schema = CubeSchema(
            name="AirQualityCube", fact_table="fact_air_quality",
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
        dim_dfs = {"Time": pd.read_csv(cube_dir / "dim_time.csv"),
                   "Location": pd.read_csv(cube_dir / "dim_location.csv"),
                   "Pollutant": pd.read_csv(cube_dir / "dim_pollutant.csv")}
    return schema, fact_df, dim_dfs

def check_hit(view, targets, strict=True):
    for t in targets:
        tl = t.get('levels', {})
        if strict:
            if view.levels == tl:
                return True
        else:
            total = len(tl)
            matches = sum(1 for d, l in tl.items() if view.levels.get(d) == l)
            if total > 0 and matches >= total * 0.66:
                return True
    return False

def compute_redundancy(views):
    if len(views) <= 1:
        return 0.0
    dups = sum(1 for i in range(1, len(views)) if any(views[i].levels == views[j].levels for j in range(i)))
    return dups / (len(views) - 1)

class LLMOnlyNav:
    """LLM-Only: uses only preference score, no data interestingness."""
    def __init__(self, schema, engine, llm_client):
        config = RecommenderConfig(lambda_data=0.0, lambda_pref=0.8, lambda_div=0.2, top_k=3)
        self.rec = NavLLMRecommender(schema, engine, llm_client, config)
    
    def select_next(self, session, utterance, measure):
        recs = self.rec.recommend(session, utterance, measure)
        if recs:
            return recs[0].view, recs[0].action
        return session.current_view, None

class NavLLMNav:
    """Full NavLLM with balanced weights."""
    def __init__(self, schema, engine, llm_client):
        config = RecommenderConfig(lambda_data=0.4, lambda_pref=0.4, lambda_div=0.2, top_k=3)
        self.rec = NavLLMRecommender(schema, engine, llm_client, config)
    
    def select_next(self, session, utterance, measure):
        recs = self.rec.recommend(session, utterance, measure)
        if recs:
            return recs[0].view, recs[0].action
        return session.current_view, None

def run_session(navigator, task, schema, engine, budget=12):
    session = Session()
    initial = CubeView.create_overview(schema)
    session.add_state(initial, task['description'])
    
    hit_s, hit_r = False, False
    steps_s, steps_r = None, None
    i_data_scores = [0.0]
    views = [initial]
    parent = engine.materialize_view(initial)
    
    for step in range(budget):
        current = session.current_view
        next_view, action = navigator.select_next(session, task['description'], task['target_measure'])
        
        if next_view is None or next_view == current:
            break
        
        try:
            i_data = engine.compute_data_interestingness(next_view, parent, task['target_measure'])
        except:
            i_data = 0.0
        i_data_scores.append(i_data)
        session.add_state(next_view, "")
        views.append(next_view)
        
        if not hit_s and check_hit(next_view, task['target_views'], strict=True):
            hit_s, steps_s = True, step + 1
        if not hit_r and check_hit(next_view, task['target_views'], strict=False):
            hit_r, steps_r = True, step + 1
    
    return {
        'hit_strict': hit_s, 'hit_relaxed': hit_r,
        'steps_to_hit_strict': steps_s, 'steps_to_hit_relaxed': steps_r,
        'cumulative_i_data': sum(i_data_scores),
        'redundancy': compute_redundancy(views),
        'num_steps': len(views) - 1
    }

def main():
    print("="*70)
    print("DIFFERENTIATED BASELINE EXPERIMENTS")
    print("20 tasks × 2 runs = 40 sessions per method")
    print("="*70)
    
    output_file = RESULTS_DIR / "differentiated_results.csv"
    
    # Check for existing results
    completed = set()
    all_results = []
    if output_file.exists():
        df = pd.read_csv(output_file)
        all_results = df.to_dict('records')
        for _, row in df.iterrows():
            completed.add((row['task_id'], row['method'], int(row['run'])))
        print(f"Resuming: {len(completed)} sessions already done")
    
    methods = ['NavLLM', 'LLM-Only', 'Heuristic-Nav', 'SeeDB', 'Greedy-Best', 'BFS', 'Random-Nav']
    tasks = load_tasks()
    num_runs = 2
    
    llm_client = DeepSeekClient(api_key=API_KEY)
    cube_cache = {}
    
    total = len(tasks) * len(methods) * num_runs
    remaining = total - len(completed)
    print(f"Total: {total}, Remaining: {remaining}")
    
    done = 0
    for task in tasks:
        cube_name = task['cube_name']
        if cube_name not in cube_cache:
            print(f"\nLoading {cube_name}...")
            cube_cache[cube_name] = load_cube(cube_name)
        schema, fact_df, dim_dfs = cube_cache[cube_name]
        engine = PandasCubeEngine(schema, fact_df, dim_dfs)
        
        for method in methods:
            for run in range(num_runs):
                if (task['task_id'], method, run) in completed:
                    continue
                
                done += 1
                print(f"[{done}/{remaining}] {task['task_id']} | {method} | run {run}...", end=' ', flush=True)
                
                try:
                    # Create navigator
                    if method == 'NavLLM':
                        nav = NavLLMNav(schema, engine, llm_client)
                    elif method == 'LLM-Only':
                        nav = LLMOnlyNav(schema, engine, llm_client)
                    elif method == 'Heuristic-Nav':
                        nav = HeuristicNav(schema, engine)
                    elif method == 'SeeDB':
                        nav = SeeDBNav(schema, engine)
                    elif method == 'Greedy-Best':
                        nav = GreedyBestNav(schema, engine)
                    elif method == 'BFS':
                        nav = BFSNav(schema, engine)
                    elif method == 'Random-Nav':
                        nav = RandomNav(schema, engine)
                    
                    start = time.time()
                    result = run_session(nav, task, schema, engine)
                    elapsed = time.time() - start
                    
                    result['task_id'] = task['task_id']
                    result['cube'] = cube_name
                    result['method'] = method
                    result['run'] = run
                    result['timestamp'] = datetime.now().isoformat()
                    all_results.append(result)
                    
                    print(f"hit={result['hit_strict']}, i_data={result['cumulative_i_data']:.2f}, {elapsed:.1f}s")
                    pd.DataFrame(all_results).to_csv(output_file, index=False)
                    
                    if method in ['NavLLM', 'LLM-Only']:
                        time.sleep(0.3)
                        
                except Exception as e:
                    print(f"ERROR: {e}")
                    import traceback
                    traceback.print_exc()
                    result = {
                        'task_id': task['task_id'], 'cube': cube_name, 'method': method, 'run': run,
                        'hit_strict': False, 'hit_relaxed': False,
                        'cumulative_i_data': 0, 'redundancy': 0, 'num_steps': 0,
                        'timestamp': datetime.now().isoformat()
                    }
                    all_results.append(result)
                    pd.DataFrame(all_results).to_csv(output_file, index=False)
    
    # Generate summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    df = pd.DataFrame(all_results)
    summary = df.groupby('method').agg({
        'hit_strict': ['mean', 'std'],
        'hit_relaxed': ['mean', 'std'],
        'cumulative_i_data': ['mean', 'std'],
        'redundancy': ['mean', 'std'],
        'num_steps': 'mean'
    }).round(3)
    print(summary)
    summary.to_csv(RESULTS_DIR / "differentiated_summary.csv")
    
    # Per-cube breakdown
    print("\nPer-cube breakdown:")
    for cube in df['cube'].unique():
        print(f"\n{cube}:")
        cube_df = df[df['cube'] == cube]
        cube_summary = cube_df.groupby('method').agg({
            'hit_strict': 'mean',
            'cumulative_i_data': 'mean'
        }).round(3)
        print(cube_summary)

if __name__ == "__main__":
    main()
