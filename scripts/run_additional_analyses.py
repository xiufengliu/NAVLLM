#!/usr/bin/env python3
"""Additional analyses: sensitivity, ablation, per-cube breakdown, significance tests."""

import os, sys, argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import from existing run_experiments.py
from run_experiments import (
    load_cube_data, create_tasks, RandomNavigator, HeuristicNavigator, 
    LLMOnlyNavigator, NavLLMNavigator, run_single_session, ExperimentConfig,
    DATA_DIR, RESULTS_DIR
)
from navllm.llm.client import DeepSeekClient
from navllm.nav.recommender import RecommenderConfig
from navllm.cube.engine import PandasCubeEngine

def run_sensitivity_analysis(llm_client, num_runs=3, budget=12):
    """Table 6: Sensitivity analysis for lambda weights."""
    print("\n" + "="*60 + "\nSENSITIVITY ANALYSIS (Table 6)\n" + "="*60)
    
    weight_configs = [
        (0.6, 0.3, 0.1, "High data"),
        (0.4, 0.4, 0.2, "Balanced"),
        (0.3, 0.5, 0.2, "High pref"),
        (0.2, 0.6, 0.2, "Very high pref"),
        (0.4, 0.3, 0.3, "High div"),
    ]
    
    tasks = create_tasks()
    results = []
    
    for ld, lp, ldiv, label in weight_configs:
        print(f"  Testing: {label} (λd={ld}, λp={lp}, λdiv={ldiv})")
        
        for task in tasks:
            try:
                schema, fact_df, dim_dfs = load_cube_data(task.cube_name)
                engine = PandasCubeEngine(schema, fact_df, dim_dfs)
                
                config = ExperimentConfig(
                    interaction_budget=budget, num_runs=num_runs,
                    lambda_data=ld, lambda_pref=lp, lambda_div=ldiv, llm_client=llm_client
                )
                rec_config = RecommenderConfig(
                    lambda_data=ld, lambda_pref=lp, lambda_div=ldiv,
                    top_k=3, generate_explanations=False
                )
                nav = NavLLMNavigator(schema, engine, rec_config, llm_client)
                
                for run in range(num_runs):
                    np.random.seed(42 + run)
                    r = run_single_session(nav, task, config, schema, engine)
                    r["lambda_data"] = ld
                    r["lambda_pref"] = lp
                    r["lambda_div"] = ldiv
                    r["config_label"] = label
                    r["task_id"] = task.task_id
                    r["cube"] = task.cube_name
                    results.append(r)
            except Exception as e:
                print(f"    Skip {task.task_id}: {e}")
    
    df = pd.DataFrame(results)
    summary = df.groupby("config_label").agg({
        "hit": "mean", "cumulative_i_data": "mean", "redundancy": "mean", "coverage": "mean"
    }).round(3)
    print("\n" + summary.to_string())
    return df

def run_ablation_study(llm_client, num_runs=5, budget=12):
    """Table 5: Ablation study."""
    print("\n" + "="*60 + "\nABLATION STUDY (Table 5)\n" + "="*60)
    
    configs = [
        ("Full NavLLM", 0.4, 0.4, 0.2),
        ("w/o I_pref (data+div)", 0.6, 0.0, 0.4),
        ("w/o I_data (pref+div)", 0.0, 0.6, 0.4),
        ("w/o I_div (data+pref)", 0.5, 0.5, 0.0),
    ]
    
    tasks = create_tasks()
    results = []
    
    for name, ld, lp, ldiv in configs:
        print(f"  Testing: {name}")
        
        for task in tasks:
            try:
                schema, fact_df, dim_dfs = load_cube_data(task.cube_name)
                engine = PandasCubeEngine(schema, fact_df, dim_dfs)
                
                config = ExperimentConfig(
                    interaction_budget=budget, num_runs=num_runs,
                    lambda_data=ld, lambda_pref=lp, lambda_div=ldiv, llm_client=llm_client
                )
                rec_config = RecommenderConfig(
                    lambda_data=ld, lambda_pref=lp, lambda_div=ldiv,
                    top_k=3, generate_explanations=False
                )
                nav = NavLLMNavigator(schema, engine, rec_config, llm_client)
                
                for run in range(num_runs):
                    np.random.seed(42 + run)
                    r = run_single_session(nav, task, config, schema, engine)
                    r["config"] = name
                    r["task_id"] = task.task_id
                    r["cube"] = task.cube_name
                    results.append(r)
            except Exception as e:
                print(f"    Skip {task.task_id}: {e}")
    
    df = pd.DataFrame(results)
    summary = df.groupby("config").agg({
        "hit": "mean", "redundancy": "mean", "coverage": "mean", "cumulative_i_data": "mean"
    }).round(3)
    print("\n" + summary.to_string())
    return df

def compute_per_cube_breakdown(results_file):
    """Table 4: Per-cube breakdown."""
    print("\n" + "="*60 + "\nPER-CUBE BREAKDOWN (Table 4)\n" + "="*60)
    
    df = pd.read_csv(results_file)
    summary = df.groupby(["cube", "method"]).agg({
        "hit": "mean", "cumulative_i_data": "mean", "redundancy": "mean"
    }).round(3)
    print("\n" + summary.to_string())
    return summary

def compute_significance_tests(results_file):
    """Table 7: Statistical significance (p-values)."""
    print("\n" + "="*60 + "\nSTATISTICAL SIGNIFICANCE (Table 7)\n" + "="*60)
    
    df = pd.read_csv(results_file)
    navllm = df[df["method"] == "NavLLM"]
    
    results = []
    for method in ["Random-Nav", "Heuristic-Nav", "LLM-Only"]:
        other = df[df["method"] == method]
        for metric in ["hit", "cumulative_i_data", "redundancy", "coverage"]:
            try:
                t_stat, p_val = stats.ttest_ind(
                    navllm[metric].dropna(), 
                    other[metric].dropna(),
                    equal_var=False  # Welch's t-test
                )
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                results.append({
                    "Comparison": f"NavLLM vs {method}",
                    "Metric": metric,
                    "t-stat": round(t_stat, 3),
                    "p-value": round(p_val, 4),
                    "Sig": sig
                })
            except Exception as e:
                print(f"  Error {method}/{metric}: {e}")
    
    sig_df = pd.DataFrame(results)
    print("\n" + sig_df.to_string(index=False))
    return sig_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepseek-key", required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--budget", type=int, default=12)
    parser.add_argument("--main-results", type=str, help="Path to main results CSV")
    args = parser.parse_args()
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    llm_client = DeepSeekClient(api_key=args.deepseek_key)
    
    # Find latest main results if not specified
    if args.main_results:
        main_results_file = Path(args.main_results)
    else:
        raw_files = sorted(RESULTS_DIR.glob("raw_results_*.csv"), reverse=True)
        if raw_files:
            main_results_file = raw_files[0]
            print(f"Using main results: {main_results_file}")
        else:
            print("No main results found. Run run_experiments.py first.")
            return
    
    # Run analyses
    sens_df = run_sensitivity_analysis(llm_client, num_runs=3, budget=args.budget)
    ablation_df = run_ablation_study(llm_client, num_runs=args.runs, budget=args.budget)
    per_cube = compute_per_cube_breakdown(main_results_file)
    sig_df = compute_significance_tests(main_results_file)
    
    # Save results
    sens_df.to_csv(RESULTS_DIR / f"sensitivity_{ts}.csv", index=False)
    ablation_df.to_csv(RESULTS_DIR / f"ablation_{ts}.csv", index=False)
    per_cube.to_csv(RESULTS_DIR / f"per_cube_{ts}.csv")
    sig_df.to_csv(RESULTS_DIR / f"significance_{ts}.csv", index=False)
    
    print(f"\nResults saved to {RESULTS_DIR}")

if __name__ == "__main__":
    main()
