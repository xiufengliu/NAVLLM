#!/usr/bin/env python3
"""Generate LaTeX tables for the paper from experiment results."""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

RESULTS_DIR = Path(__file__).parent.parent / "results"

def load_main_results():
    """Load combined main results."""
    f = RESULTS_DIR / "combined_main_results.csv"
    if f.exists():
        return pd.read_csv(f)
    return None

def load_ablation():
    """Load ablation results."""
    f = RESULTS_DIR / "ablation_final.csv"
    if f.exists():
        return pd.read_csv(f)
    return None

def load_sensitivity():
    """Load sensitivity results."""
    f = RESULTS_DIR / "sensitivity_final.csv"
    if f.exists():
        return pd.read_csv(f)
    return None

def generate_main_table(df):
    """Generate Table 3: Navigation quality results."""
    print("\n" + "="*70)
    print("TABLE 3: Navigation Quality Results")
    print("="*70)
    
    summary = df.groupby('method').agg({
        'hit_strict': ['mean', 'std'],
        'hit_relaxed': ['mean', 'std'],
        'cumulative_i_data': ['mean', 'std']
    })
    
    # Order methods
    order = ['LLM-Only', 'NavLLM', 'Heuristic-Nav', 'SeeDB', 'Greedy-Best', 'BFS', 'Random-Nav']
    
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Navigation quality results (mean $\\pm$ std across 20 tasks $\\times$ 2 runs = 40 sessions per method).}")
    print("\\label{tab:quality_results}")
    print("\\begin{tabular}{lccc}")
    print("\\toprule")
    print("\\textbf{Method} & \\textbf{Hit Rate@12} & \\textbf{Hit Rate@12} & \\textbf{Cumulative $I_{\\text{data}}$} \\\\")
    print(" & \\textbf{(strict)} & \\textbf{(relaxed)} & \\\\")
    print("\\midrule")
    
    for method in order:
        if method in summary.index:
            row = summary.loc[method]
            hs = f"{row[('hit_strict', 'mean')]:.3f} $\\pm$ {row[('hit_strict', 'std')]:.2f}"
            hr = f"{row[('hit_relaxed', 'mean')]:.3f} $\\pm$ {row[('hit_relaxed', 'std')]:.2f}"
            idata = f"{row[('cumulative_i_data', 'mean')]:.2f} $\\pm$ {row[('cumulative_i_data', 'std')]:.2f}"
            
            # Bold best values
            name = "\\textsc{NavLLM}" if method == "NavLLM" else method
            print(f"{name} & {hs} & {hr} & {idata} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

def generate_efficiency_table(df):
    """Generate Table 4: Efficiency results."""
    print("\n" + "="*70)
    print("TABLE 4: Efficiency Results")
    print("="*70)
    
    summary = df.groupby('method').agg({
        'redundancy': ['mean', 'std'],
        'num_steps': 'mean'
    })
    
    order = ['Heuristic-Nav', 'SeeDB', 'Greedy-Best', 'BFS', 'Random-Nav', 'NavLLM', 'LLM-Only']
    
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Efficiency results (mean $\\pm$ std across 40 sessions per method).}")
    print("\\label{tab:efficiency_results}")
    print("\\begin{tabular}{lcc}")
    print("\\toprule")
    print("\\textbf{Method} & \\textbf{Redundancy (\\%)} & \\textbf{Avg Steps} \\\\")
    print("\\midrule")
    
    for method in order:
        if method in summary.index:
            row = summary.loc[method]
            red = row[('redundancy', 'mean')] * 100
            red_std = row[('redundancy', 'std')] * 100
            steps = row[('num_steps', 'mean')]
            
            name = "\\textsc{NavLLM}" if method == "NavLLM" else method
            print(f"{name} & {red:.1f} $\\pm$ {red_std:.1f} & {steps:.1f} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

def generate_per_cube_table(df):
    """Generate Table 5: Per-cube breakdown."""
    print("\n" + "="*70)
    print("TABLE 5: Per-Cube Breakdown")
    print("="*70)
    
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Per-cube breakdown of navigation quality metrics (strict hit rate).}")
    print("\\label{tab:per_cube}")
    print("\\begin{tabular}{llcc}")
    print("\\toprule")
    print("\\textbf{Cube} & \\textbf{Method} & \\textbf{Hit Rate} & \\textbf{Cum. $I_{\\text{data}}$} \\\\")
    print("\\midrule")
    
    cube_map = {'m5': 'SalesCube', 'manufacturing': 'Manufacturing', 'air_quality': 'AirQuality'}
    
    for cube in ['m5', 'manufacturing', 'air_quality']:
        cube_df = df[df['cube'] == cube]
        summary = cube_df.groupby('method').agg({
            'hit_strict': 'mean',
            'cumulative_i_data': 'mean'
        }).sort_values('hit_strict', ascending=False)
        
        # Show top 4 methods
        for i, (method, row) in enumerate(summary.head(4).iterrows()):
            prefix = f"\\multirow{{4}}{{*}}{{{cube_map[cube]}}}" if i == 0 else ""
            name = "\\textsc{NavLLM}" if method == "NavLLM" else method
            bold_hr = "\\textbf{" + f"{row['hit_strict']:.3f}" + "}" if i == 0 else f"{row['hit_strict']:.3f}"
            bold_id = "\\textbf{" + f"{row['cumulative_i_data']:.2f}" + "}" if row['cumulative_i_data'] == summary['cumulative_i_data'].max() else f"{row['cumulative_i_data']:.2f}"
            print(f"{prefix} & {name} & {bold_hr} & {bold_id} \\\\")
        print("\\midrule")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

def generate_significance_table(df):
    """Generate Table 6: Statistical significance."""
    print("\n" + "="*70)
    print("TABLE 6: Statistical Significance")
    print("="*70)
    
    navllm = df[df['method'] == 'NavLLM']
    
    comparisons = [
        ('LLM-Only', 'hit_strict', 'NavLLM < LLM-Only'),
        ('LLM-Only', 'cumulative_i_data', 'NavLLM > LLM-Only'),
        ('Heuristic-Nav', 'hit_strict', 'NavLLM = Heuristic'),
        ('Heuristic-Nav', 'cumulative_i_data', 'NavLLM > Heuristic'),
        ('Random-Nav', 'hit_strict', 'NavLLM > Random'),
        ('Random-Nav', 'cumulative_i_data', 'NavLLM > Random'),
    ]
    
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Statistical significance of \\textsc{NavLLM} vs baselines (Welch's $t$-test).}")
    print("\\label{tab:significance}")
    print("\\begin{tabular}{llcc}")
    print("\\toprule")
    print("\\textbf{Comparison} & \\textbf{Metric} & \\textbf{Direction} & \\textbf{$p$-value} \\\\")
    print("\\midrule")
    
    for baseline, metric, direction in comparisons:
        baseline_data = df[df['method'] == baseline][metric]
        navllm_data = navllm[metric]
        
        t_stat, p_val = stats.ttest_ind(navllm_data, baseline_data, equal_var=False)
        
        if p_val < 0.001:
            p_str = "$< 0.001^{***}$"
        elif p_val < 0.01:
            p_str = f"${p_val:.3f}^{{**}}$"
        elif p_val < 0.05:
            p_str = f"${p_val:.3f}^{{*}}$"
        else:
            p_str = f"${p_val:.3f}$"
        
        metric_name = "Hit Rate" if metric == "hit_strict" else "Cum. $I_{\\text{data}}$"
        print(f"vs {baseline} & {metric_name} & {direction} & {p_str} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

def generate_ablation_table(df):
    """Generate Table 7: Ablation study."""
    print("\n" + "="*70)
    print("TABLE 7: Ablation Study")
    print("="*70)
    
    summary = df.groupby('config').agg({
        'hit_strict': 'mean',
        'redundancy': 'mean',
        'coverage': 'mean',
        'cumulative_i_data': 'mean'
    }).round(2)
    
    n_sessions = len(df) // len(summary)
    
    print("\\begin{table}[t]")
    print("\\centering")
    print(f"\\caption{{Ablation study: impact of utility function components (20 tasks $\\times$ 2 runs = {n_sessions} sessions per config).}}")
    print("\\label{tab:ablation}")
    print("\\begin{tabular}{lcccc}")
    print("\\toprule")
    print("\\textbf{Configuration} & \\textbf{Hit Rate} & \\textbf{Redundancy} & \\textbf{Coverage} & \\textbf{Cum. $I_{\\text{data}}$} \\\\")
    print("\\midrule")
    
    order = ['Full', 'w/o I_pref', 'w/o I_data', 'w/o I_div']
    for config in order:
        if config in summary.index:
            row = summary.loc[config]
            name = "Full \\textsc{NavLLM}" if config == "Full" else config
            print(f"{name} & {row['hit_strict']:.2f} & {row['redundancy']:.2f} & {row['coverage']:.2f} & {row['cumulative_i_data']:.2f} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

def generate_sensitivity_table(df):
    """Generate Table 8: Sensitivity analysis."""
    print("\n" + "="*70)
    print("TABLE 8: Sensitivity Analysis")
    print("="*70)
    
    summary = df.groupby(['lambda_data', 'lambda_pref', 'lambda_div']).agg({
        'hit_strict': 'mean',
        'cumulative_i_data': 'mean',
        'redundancy': 'mean'
    }).round(2)
    
    n_sessions = len(df) // len(summary)
    
    print("\\begin{table}[t]")
    print("\\centering")
    print(f"\\caption{{Sensitivity analysis: impact of utility function weights (20 tasks $\\times$ 2 runs = {n_sessions} sessions per config).}}")
    print("\\label{tab:sensitivity}")
    print("\\begin{tabular}{ccccccc}")
    print("\\toprule")
    print("$\\lambda_{\\text{data}}$ & $\\lambda_{\\text{pref}}$ & $\\lambda_{\\text{div}}$ & \\textbf{Hit Rate} & \\textbf{Cum. $I_{\\text{data}}$} & \\textbf{Redundancy} \\\\")
    print("\\midrule")
    
    for (ld, lp, ldiv), row in summary.iterrows():
        print(f"{ld} & {lp} & {ldiv} & {row['hit_strict']:.2f} & {row['cumulative_i_data']:.2f} & {row['redundancy']:.2f} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

def main():
    print("="*70)
    print("GENERATING PAPER TABLES")
    print("="*70)
    
    # Main results
    main_df = load_main_results()
    if main_df is not None:
        print(f"\nMain results: {len(main_df)} rows")
        generate_main_table(main_df)
        generate_efficiency_table(main_df)
        generate_per_cube_table(main_df)
        generate_significance_table(main_df)
    else:
        print("Main results not found!")
    
    # Ablation
    abl_df = load_ablation()
    if abl_df is not None:
        print(f"\nAblation results: {len(abl_df)} rows")
        generate_ablation_table(abl_df)
    else:
        print("\nAblation results not found - experiments still running")
    
    # Sensitivity
    sens_df = load_sensitivity()
    if sens_df is not None:
        print(f"\nSensitivity results: {len(sens_df)} rows")
        generate_sensitivity_table(sens_df)
    else:
        print("\nSensitivity results not found - experiments still running")

if __name__ == "__main__":
    main()
