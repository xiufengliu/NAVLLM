# NavLLM

NavLLM is a system for LLM-assisted conversational navigation over multidimensional data cubes. It helps analysts explore OLAP cubes more effectively by combining data-driven interestingness with natural language understanding.

## Overview

When exploring data cubes, analysts often struggle to decide which drill-down or slice operation to perform next. NavLLM addresses this by recommending the most promising views based on:

- **Data interestingness**: Statistical measures like deviation from parent aggregates
- **User preferences**: Understanding what the analyst is looking for from conversation context  
- **Exploration diversity**: Avoiding redundant views that show similar information

The key design principle is that the LLM handles semantic understanding (interpreting user intent, generating explanations) while a conventional cube engine handles all numerical computations. This keeps results verifiable and avoids hallucination issues.

## Installation

```bash
git clone https://github.com/xiufengliu/NAVLLM.git
cd NAVLLM
pip install -e .
```

Set up your LLM API key:
```bash
export DEEPSEEK_API_KEY=your_key_here
# or
export OPENAI_API_KEY=your_key_here
```

## Quick Start

```python
from navllm.cube import PandasCubeEngine
from navllm.nav import NavLLMRecommender

# Load your cube
engine = PandasCubeEngine.from_csv(
    fact_path="data/processed/m5/fact_sales.csv",
    dim_paths={
        "Time": "data/processed/m5/dim_time.csv",
        "Product": "data/processed/m5/dim_product.csv",
        "Store": "data/processed/m5/dim_store.csv"
    }
)

# Create recommender
recommender = NavLLMRecommender(
    cube_engine=engine,
    weights={"data": 0.4, "pref": 0.4, "div": 0.2}
)

# Get recommendations
recs = recommender.recommend(
    current_view=engine.get_overview(),
    utterance="Show me where sales are declining",
    top_k=3
)

for rec in recs:
    print(f"{rec.description}: {rec.explanation}")
```

## Datasets

The repository includes three cubes for experiments:

| Cube | Domain | Dimensions | Measures | Rows |
|------|--------|------------|----------|------|
| SalesCube | Retail | Time, Product, Store | units, revenue | ~1M |
| ManufacturingCube | Operations | Time, Line, Product | throughput, defects | ~394K |
| AirQualityCube | Environment | Time, Location, Pollutant | concentration, AQI | ~2.5M |

Download and preprocess:
```bash
python scripts/download_data.py
python scripts/preprocess_data.py
```

## Running Experiments

```bash
# Run main experiments
python scripts/run_experiments.py

# Run ablation study
python scripts/run_ablation_sensitivity.py

# Generate result tables
python scripts/generate_paper_tables.py
```

## Project Structure

```
navllm/
├── src/navllm/
│   ├── cube/       # Cube engine, views, actions
│   ├── llm/        # LLM client and preference scorer
│   ├── nav/        # Navigation session and recommender
│   └── eval/       # Evaluation metrics
├── scripts/        # Experiment scripts
├── configs/        # Cube configurations and tasks
├── data/           # Datasets (after download)
└── tests/          # Unit tests
```

## Configuration

Utility function weights can be tuned based on your needs:

- Higher `data` weight → more focus on statistical anomalies
- Higher `pref` weight → more focus on user's stated goals
- Higher `div` weight → broader exploration, less redundancy

Default weights (0.4, 0.4, 0.2) work well for most cases.

## Requirements

- Python 3.10+
- pandas, numpy
- openai or deepseek SDK
- See `pyproject.toml` for full list

## License

MIT License

## Contact

Xiufeng Liu - xiuli@dtu.dk
