#!/usr/bin/env python3
"""
Preprocess raw data into cube-ready format.

Creates fact tables and dimension tables for each cube.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


def preprocess_air_quality():
    """
    Preprocess Beijing Air Quality data into AirQualityCube format.
    """
    print("\n" + "="*60)
    print("Preprocessing Air Quality Data")
    print("="*60)
    
    aq_dir = RAW_DIR / "air_quality" / "PRSA_Data_20130301-20170228"
    output_dir = PROCESSED_DIR / "air_quality"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all station files
    all_data = []
    for csv_file in aq_dir.glob("*.csv"):
        station_name = csv_file.stem.replace("PRSA_Data_", "").replace("_20130301-20170228", "")
        df = pd.read_csv(csv_file)
        df['station'] = station_name
        all_data.append(df)
    
    if not all_data:
        print("No air quality data found. Please download first.")
        return
    
    df = pd.concat(all_data, ignore_index=True)
    print(f"Loaded {len(df):,} records from {len(all_data)} stations")
    
    # Create time dimension columns
    df['date'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df['week'] = df['date'].dt.isocalendar().week
    df['season'] = df['month'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    })
    
    # Create location hierarchy (station -> district -> city -> region)
    # For Beijing data, we'll create a simple hierarchy
    district_map = {
        'Dongsi': 'Dongcheng', 'Tiantan': 'Dongcheng',
        'Guanyuan': 'Xicheng', 'Wanshouxigong': 'Xicheng',
        'Aotizhongxin': 'Chaoyang', 'Nongzhanguan': 'Chaoyang',
        'Wanliu': 'Haidian', 'Gucheng': 'Shijingshan',
        'Changping': 'Changping', 'Dingling': 'Changping',
        'Huairou': 'Huairou', 'Shunyi': 'Shunyi'
    }
    df['district'] = df['station'].map(district_map).fillna('Other')
    df['city'] = 'Beijing'
    df['region'] = 'North China'
    
    # Melt pollutant columns into rows
    pollutant_cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
    
    # Create fact table with pollutant as dimension
    fact_records = []
    for _, row in df.iterrows():
        for pollutant in pollutant_cols:
            if pd.notna(row.get(pollutant)):
                fact_records.append({
                    'time_id': row['date'],
                    'station_id': row['station'],
                    'pollutant_id': pollutant,
                    'concentration': row[pollutant],
                    'temperature': row.get('TEMP'),
                    'pressure': row.get('PRES'),
                    'humidity': row.get('DEWP'),
                    'wind_speed': row.get('WSPM'),
                })
    
    fact_df = pd.DataFrame(fact_records)
    
    # Add time dimension columns to fact
    fact_df['hour'] = pd.to_datetime(fact_df['time_id']).dt.hour
    fact_df['day'] = pd.to_datetime(fact_df['time_id']).dt.date
    fact_df['week'] = pd.to_datetime(fact_df['time_id']).dt.isocalendar().week
    fact_df['month'] = pd.to_datetime(fact_df['time_id']).dt.month
    fact_df['year'] = pd.to_datetime(fact_df['time_id']).dt.year
    fact_df['season'] = fact_df['month'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    })
    fact_df['all_time'] = 'All'
    
    # Add location dimension columns
    fact_df['district'] = fact_df['station_id'].map(district_map).fillna('Other')
    fact_df['city'] = 'Beijing'
    fact_df['region'] = 'North China'
    fact_df['all_location'] = 'All'
    fact_df['all_pollutant'] = 'All'
    
    # Save fact table
    fact_df.to_csv(output_dir / "fact_air_quality.csv", index=False)
    print(f"Saved fact table: {len(fact_df):,} records")
    
    # Create dimension tables
    dim_time = fact_df[['time_id', 'hour', 'day', 'week', 'month', 'year', 'season', 'all_time']].drop_duplicates()
    dim_time.to_csv(output_dir / "dim_time.csv", index=False)
    
    dim_location = fact_df[['station_id', 'district', 'city', 'region', 'all_location']].drop_duplicates()
    dim_location.to_csv(output_dir / "dim_location.csv", index=False)
    
    dim_pollutant = fact_df[['pollutant_id', 'all_pollutant']].drop_duplicates()
    dim_pollutant.to_csv(output_dir / "dim_pollutant.csv", index=False)
    
    print(f"Created dimension tables")
    print(f"  Time: {len(dim_time):,} entries")
    print(f"  Location: {len(dim_location):,} entries")
    print(f"  Pollutant: {len(dim_pollutant):,} entries")
    
    return output_dir


def preprocess_manufacturing():
    """
    Preprocess manufacturing data into ManufacturingCube format.
    """
    print("\n" + "="*60)
    print("Preprocessing Manufacturing Data")
    print("="*60)
    
    mfg_file = RAW_DIR / "manufacturing" / "manufacturing_data.csv"
    output_dir = PROCESSED_DIR / "manufacturing"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not mfg_file.exists():
        print("Manufacturing data not found. Please run download_data.py first.")
        return
    
    df = pd.read_csv(mfg_file)
    print(f"Loaded {len(df):,} records")
    
    # Add 'All' columns for hierarchies
    df['all_time'] = 'All'
    df['all_line'] = 'All'
    df['all_product'] = 'All'
    df['category_id'] = df['family_id'].str.split('_').str[0]
    
    # Save fact table
    fact_cols = ['date', 'plant_id', 'line_id', 'machine_id', 'shift', 
                 'family_id', 'variant_id', 'category_id',
                 'throughput', 'defect_count', 'defect_rate',
                 'day', 'week', 'month', 'year', 'all_time',
                 'all_line', 'all_product']
    
    fact_df = df[fact_cols].copy()
    fact_df.to_csv(output_dir / "fact_production.csv", index=False)
    print(f"Saved fact table: {len(fact_df):,} records")
    
    # Create dimension tables
    dim_time = df[['date', 'day', 'week', 'month', 'year', 'all_time']].drop_duplicates()
    dim_time.to_csv(output_dir / "dim_time.csv", index=False)
    
    dim_line = df[['machine_id', 'line_id', 'plant_id', 'all_line']].drop_duplicates()
    dim_line.to_csv(output_dir / "dim_line.csv", index=False)
    
    dim_product = df[['variant_id', 'family_id', 'category_id', 'all_product']].drop_duplicates()
    dim_product.to_csv(output_dir / "dim_product.csv", index=False)
    
    dim_shift = pd.DataFrame({'shift': df['shift'].unique()})
    dim_shift.to_csv(output_dir / "dim_shift.csv", index=False)
    
    print(f"Created dimension tables")
    print(f"  Time: {len(dim_time):,} entries")
    print(f"  Line: {len(dim_line):,} entries")
    print(f"  Product: {len(dim_product):,} entries")
    print(f"  Shift: {len(dim_shift):,} entries")
    
    return output_dir


def preprocess_m5():
    """
    Preprocess M5 data into SalesCube format.
    
    Requires M5 data to be downloaded from Kaggle.
    """
    print("\n" + "="*60)
    print("Preprocessing M5 Sales Data")
    print("="*60)
    
    m5_dir = RAW_DIR / "m5"
    output_dir = PROCESSED_DIR / "m5"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for required files
    sales_file = m5_dir / "sales_train_validation.csv"
    calendar_file = m5_dir / "calendar.csv"
    prices_file = m5_dir / "sell_prices.csv"
    
    if not sales_file.exists():
        print(f"M5 data not found at {m5_dir}")
        print("Please download from: https://www.kaggle.com/c/m5-forecasting-accuracy/data")
        print("\nCreating sample M5-like data for testing...")
        create_sample_m5_data(output_dir)
        return output_dir
    
    print("Loading M5 data (this may take a while)...")
    
    # Load calendar
    calendar = pd.read_csv(calendar_file)
    
    # Load sales (wide format)
    sales = pd.read_csv(sales_file)
    
    # Melt to long format
    id_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    day_cols = [c for c in sales.columns if c.startswith('d_')]
    
    sales_long = sales.melt(
        id_vars=id_cols,
        value_vars=day_cols,
        var_name='d',
        value_name='units'
    )
    
    # Merge with calendar
    sales_long = sales_long.merge(calendar[['d', 'date', 'wm_yr_wk', 'month', 'year']], on='d')
    
    # Add time dimensions
    sales_long['date'] = pd.to_datetime(sales_long['date'])
    sales_long['day'] = sales_long['date'].dt.date
    sales_long['week'] = sales_long['date'].dt.isocalendar().week
    sales_long['quarter'] = sales_long['date'].dt.quarter
    sales_long['all_time'] = 'All'
    sales_long['all_product'] = 'All'
    sales_long['all_store'] = 'All'
    
    # Estimate revenue (if prices available)
    if prices_file.exists():
        prices = pd.read_csv(prices_file)
        sales_long = sales_long.merge(
            prices, 
            on=['store_id', 'item_id', 'wm_yr_wk'], 
            how='left'
        )
        sales_long['revenue'] = sales_long['units'] * sales_long['sell_price'].fillna(10)
    else:
        sales_long['revenue'] = sales_long['units'] * 10  # Default price
    
    # Save fact table
    fact_df = sales_long[['date', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id',
                          'units', 'revenue', 'day', 'week', 'month', 'quarter', 'year',
                          'all_time', 'all_product', 'all_store']]
    fact_df.to_csv(output_dir / "fact_sales.csv", index=False)
    print(f"Saved fact table: {len(fact_df):,} records")
    
    # Create dimension tables
    dim_time = fact_df[['date', 'day', 'week', 'month', 'quarter', 'year', 'all_time']].drop_duplicates()
    dim_time.to_csv(output_dir / "dim_time.csv", index=False)
    
    dim_product = fact_df[['item_id', 'dept_id', 'cat_id', 'all_product']].drop_duplicates()
    dim_product.to_csv(output_dir / "dim_product.csv", index=False)
    
    dim_store = fact_df[['store_id', 'state_id', 'all_store']].drop_duplicates()
    dim_store.to_csv(output_dir / "dim_store.csv", index=False)
    
    print(f"Created dimension tables")
    
    return output_dir


def create_sample_m5_data(output_dir):
    """Create sample M5-like data for testing when real data is not available."""
    np.random.seed(42)
    
    # Configuration matching M5 structure
    stores = ['CA_1', 'CA_2', 'CA_3', 'CA_4', 'TX_1', 'TX_2', 'TX_3', 'WI_1', 'WI_2', 'WI_3']
    states = {s: s.split('_')[0] for s in stores}
    categories = ['FOODS', 'HOUSEHOLD', 'HOBBIES']
    departments = {
        'FOODS': ['FOODS_1', 'FOODS_2', 'FOODS_3'],
        'HOUSEHOLD': ['HOUSEHOLD_1', 'HOUSEHOLD_2'],
        'HOBBIES': ['HOBBIES_1', 'HOBBIES_2']
    }
    
    # Generate 100 items per department
    items = []
    for cat, depts in departments.items():
        for dept in depts:
            for i in range(20):  # 20 items per dept for manageable size
                items.append({'item_id': f"{dept}_{i:03d}", 'dept_id': dept, 'cat_id': cat})
    
    items_df = pd.DataFrame(items)
    
    # Date range: 2 years
    dates = pd.date_range('2022-01-01', '2023-12-31', freq='D')
    
    records = []
    for date in dates:
        for store in stores:
            for _, item in items_df.iterrows():
                # Base sales
                base = 5 + np.random.exponential(3)
                
                # Seasonal effects
                if date.month in [11, 12]:
                    base *= 1.5
                elif date.month in [1, 2]:
                    base *= 0.7
                
                # Store effects
                if store.startswith('CA'):
                    base *= 1.3
                
                # Category effects
                if item['cat_id'] == 'FOODS':
                    base *= 2
                
                units = max(0, int(base))
                price = np.random.uniform(3, 20)
                
                records.append({
                    'date': date,
                    'item_id': item['item_id'],
                    'dept_id': item['dept_id'],
                    'cat_id': item['cat_id'],
                    'store_id': store,
                    'state_id': states[store],
                    'units': units,
                    'revenue': units * price,
                    'day': date.date(),
                    'week': date.isocalendar()[1],
                    'month': date.month,
                    'quarter': date.quarter,
                    'year': date.year,
                    'all_time': 'All',
                    'all_product': 'All',
                    'all_store': 'All',
                })
    
    fact_df = pd.DataFrame(records)
    fact_df.to_csv(output_dir / "fact_sales.csv", index=False)
    print(f"Created sample fact table: {len(fact_df):,} records")
    
    # Dimension tables
    dim_time = fact_df[['date', 'day', 'week', 'month', 'quarter', 'year', 'all_time']].drop_duplicates()
    dim_time.to_csv(output_dir / "dim_time.csv", index=False)
    
    dim_product = fact_df[['item_id', 'dept_id', 'cat_id', 'all_product']].drop_duplicates()
    dim_product.to_csv(output_dir / "dim_product.csv", index=False)
    
    dim_store = fact_df[['store_id', 'state_id', 'all_store']].drop_duplicates()
    dim_store.to_csv(output_dir / "dim_store.csv", index=False)
    
    print("Created dimension tables")


def main():
    print("NavLLM Data Preprocessing")
    print("="*60)
    
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    preprocess_air_quality()
    preprocess_manufacturing()
    preprocess_m5()
    
    print("\n" + "="*60)
    print("Preprocessing complete!")
    print("="*60)
    
    # Show summary
    print("\nProcessed data summary:")
    for cube_dir in PROCESSED_DIR.iterdir():
        if cube_dir.is_dir():
            files = list(cube_dir.glob("*.csv"))
            total_size = sum(f.stat().st_size for f in files) / (1024*1024)
            print(f"  {cube_dir.name}: {len(files)} files, {total_size:.1f} MB")


if __name__ == "__main__":
    main()
