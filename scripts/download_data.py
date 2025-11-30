#!/usr/bin/env python3
"""
Download datasets for NavLLM experiments.

Datasets:
1. M5 Forecasting (SalesCube) - Kaggle competition data
2. Air Quality (AirQualityCube) - UCI/OpenAQ data  
3. Manufacturing (ManufacturingCube) - Synthetic data generation

Usage:
    python scripts/download_data.py --dataset all
    python scripts/download_data.py --dataset m5
    python scripts/download_data.py --dataset air_quality
    python scripts/download_data.py --dataset manufacturing
"""

import os
import sys
import argparse
import urllib.request
import zipfile
import gzip
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


def download_file(url: str, dest: Path, desc: str = ""):
    """Download a file with progress indicator."""
    print(f"Downloading {desc or url}...")
    
    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size) if total_size > 0 else 0
        sys.stdout.write(f"\r  Progress: {percent}%")
        sys.stdout.flush()
    
    try:
        urllib.request.urlretrieve(url, dest, progress_hook)
        print(f"\n  Saved to {dest}")
        return True
    except Exception as e:
        print(f"\n  Error: {e}")
        return False


def download_m5_data():
    """
    Download M5 Forecasting dataset.
    
    The M5 dataset requires Kaggle authentication. 
    Options:
    1. Use kaggle CLI: kaggle competitions download -c m5-forecasting-accuracy
    2. Download manually from: https://www.kaggle.com/c/m5-forecasting-accuracy/data
    3. Use alternative mirror (if available)
    """
    m5_dir = RAW_DIR / "m5"
    m5_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("M5 Forecasting Dataset (SalesCube)")
    print("="*60)
    
    # Check if kaggle is available
    kaggle_available = shutil.which("kaggle") is not None
    
    if kaggle_available:
        print("\nUsing Kaggle CLI to download M5 data...")
        print("Make sure you have accepted the competition rules at:")
        print("https://www.kaggle.com/c/m5-forecasting-accuracy/rules")
        
        os.system(f"kaggle competitions download -c m5-forecasting-accuracy -p {m5_dir}")
        
        # Extract if zip exists
        zip_file = m5_dir / "m5-forecasting-accuracy.zip"
        if zip_file.exists():
            print(f"\nExtracting {zip_file}...")
            with zipfile.ZipFile(zip_file, 'r') as zf:
                zf.extractall(m5_dir)
            print("  Extraction complete!")
    else:
        print("\nKaggle CLI not found. Please install it:")
        print("  pip install kaggle")
        print("\nOr download manually from:")
        print("  https://www.kaggle.com/c/m5-forecasting-accuracy/data")
        print(f"\nPlace the following files in: {m5_dir}")
        print("  - calendar.csv")
        print("  - sales_train_validation.csv")
        print("  - sell_prices.csv")
        print("  - sample_submission.csv")
        
        # Create placeholder instructions
        readme = m5_dir / "README.txt"
        with open(readme, 'w') as f:
            f.write("""M5 Forecasting Dataset

Download from: https://www.kaggle.com/c/m5-forecasting-accuracy/data

Required files:
- calendar.csv (contains date information)
- sales_train_validation.csv (contains sales data)
- sell_prices.csv (contains price information)

After downloading, run the preprocessing script:
  python scripts/preprocess_m5.py
""")
        print(f"\n  Created {readme}")
    
    return m5_dir


def download_air_quality_data():
    """
    Download Air Quality dataset from UCI ML Repository.
    
    Using the Beijing Multi-Site Air-Quality Data Set.
    """
    aq_dir = RAW_DIR / "air_quality"
    aq_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("Air Quality Dataset (AirQualityCube)")
    print("="*60)
    
    # UCI Beijing Air Quality dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00501/PRSA2017_Data_20130301-20170228.zip"
    zip_file = aq_dir / "air_quality.zip"
    
    if download_file(url, zip_file, "Beijing Air Quality Data"):
        print(f"\nExtracting {zip_file}...")
        try:
            with zipfile.ZipFile(zip_file, 'r') as zf:
                zf.extractall(aq_dir)
            print("  Extraction complete!")
        except Exception as e:
            print(f"  Extraction error: {e}")
    
    return aq_dir


def generate_manufacturing_data():
    """
    Generate synthetic manufacturing data.
    
    Creates realistic production data with:
    - Multiple production lines and machines
    - Shift patterns
    - Product variants
    - Defect rates with anomalies
    """
    import pandas as pd
    import numpy as np
    
    mfg_dir = RAW_DIR / "manufacturing"
    mfg_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("Manufacturing Dataset (ManufacturingCube) - Synthetic")
    print("="*60)
    
    np.random.seed(42)
    
    # Configuration
    plants = ['Plant_A', 'Plant_B']
    lines_per_plant = {'Plant_A': ['Line_1', 'Line_2', 'Line_3'], 
                       'Plant_B': ['Line_4', 'Line_5']}
    machines_per_line = 3
    shifts = ['Morning', 'Afternoon', 'Night']
    product_families = ['Family_X', 'Family_Y', 'Family_Z']
    variants_per_family = 4
    
    # Date range: 2 years of data
    dates = pd.date_range('2022-01-01', '2023-12-31', freq='D')
    
    records = []
    
    for date in dates:
        for plant in plants:
            for line in lines_per_plant[plant]:
                for machine_idx in range(1, machines_per_line + 1):
                    machine = f"{line}_M{machine_idx}"
                    for shift in shifts:
                        for family in product_families:
                            for var_idx in range(1, variants_per_family + 1):
                                variant = f"{family}_V{var_idx}"
                                
                                # Base throughput
                                base_throughput = 100 + np.random.normal(0, 10)
                                
                                # Plant effect
                                if plant == 'Plant_A':
                                    base_throughput *= 1.1
                                
                                # Shift effect
                                if shift == 'Night':
                                    base_throughput *= 0.9
                                
                                # Machine degradation over time
                                days_since_start = (date - dates[0]).days
                                degradation = 1 - (days_since_start / 1000) * 0.1
                                base_throughput *= max(degradation, 0.8)
                                
                                # Seasonal maintenance (lower in summer)
                                if date.month in [6, 7, 8]:
                                    base_throughput *= 0.95
                                
                                throughput = max(0, int(base_throughput))
                                
                                # Defect rate
                                base_defect_rate = 0.02 + np.random.normal(0, 0.005)
                                
                                # Anomaly: Line_2 has elevated defects in Q3 2023
                                if line == 'Line_2' and date.year == 2023 and date.quarter == 3:
                                    base_defect_rate *= 3
                                
                                # Night shift has higher defects
                                if shift == 'Night':
                                    base_defect_rate *= 1.3
                                
                                defect_rate = max(0, min(1, base_defect_rate))
                                defect_count = int(throughput * defect_rate)
                                
                                records.append({
                                    'date': date,
                                    'plant_id': plant,
                                    'line_id': line,
                                    'machine_id': machine,
                                    'shift': shift,
                                    'family_id': family,
                                    'variant_id': variant,
                                    'throughput': throughput,
                                    'defect_count': defect_count,
                                    'defect_rate': defect_rate,
                                    # Time dimensions
                                    'day': date.date(),
                                    'week': date.isocalendar()[1],
                                    'month': date.month,
                                    'year': date.year,
                                })
    
    df = pd.DataFrame(records)
    
    # Save to CSV
    output_file = mfg_dir / "manufacturing_data.csv"
    df.to_csv(output_file, index=False)
    print(f"\nGenerated {len(df):,} records")
    print(f"Saved to {output_file}")
    
    # Print summary
    print(f"\nData summary:")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Plants: {df['plant_id'].nunique()}")
    print(f"  Lines: {df['line_id'].nunique()}")
    print(f"  Machines: {df['machine_id'].nunique()}")
    print(f"  Product variants: {df['variant_id'].nunique()}")
    print(f"  Total throughput: {df['throughput'].sum():,}")
    print(f"  Total defects: {df['defect_count'].sum():,}")
    print(f"  Avg defect rate: {df['defect_rate'].mean():.2%}")
    
    return mfg_dir


def main():
    parser = argparse.ArgumentParser(description="Download NavLLM experiment datasets")
    parser.add_argument("--dataset", choices=["all", "m5", "air_quality", "manufacturing"],
                        default="all", help="Which dataset to download")
    args = parser.parse_args()
    
    # Ensure directories exist
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    print("NavLLM Data Download Script")
    print("="*60)
    print(f"Data directory: {DATA_DIR}")
    
    if args.dataset in ["all", "m5"]:
        download_m5_data()
    
    if args.dataset in ["all", "air_quality"]:
        download_air_quality_data()
    
    if args.dataset in ["all", "manufacturing"]:
        generate_manufacturing_data()
    
    print("\n" + "="*60)
    print("Download complete!")
    print("="*60)
    print(f"\nNext steps:")
    print("1. For M5 data: Ensure files are downloaded from Kaggle")
    print("2. Run preprocessing: python scripts/preprocess_data.py")
    print("3. Run experiments: python scripts/run_experiments.py")


if __name__ == "__main__":
    main()
