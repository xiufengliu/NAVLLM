# NavLLM Source Code

LLM-Assisted Conversational Navigation over Multidimensional Data Cubes

## Project Structure

```
src/navllm/
├── __init__.py          # Package initialization
├── cube/                # Cube data structures and engine
│   ├── schema.py        # CubeSchema, Dimension, Measure, Level (Def 1)
│   ├── view.py          # CubeView, Filter (Def 3)
│   ├── actions.py       # NavigationAction, DrillDown, RollUp, Slice (Def 4)
│   └── engine.py        # CubeEngine, PandasCubeEngine
├── llm/                 # LLM integration
│   ├── client.py        # LLMClient, OpenAIClient, MockLLMClient
│   └── scorer.py        # PreferenceScorer, ExplanationGenerator
├── nav/                 # Navigation and recommendation
│   ├── session.py       # Session, SessionState (Def 5)
│   └── recommender.py   # NavLLMRecommender (Algorithm 1)
├── eval/                # Evaluation
│   └── metrics.py       # MetricsCalculator, EvaluationResult
└── api/                 # REST API (optional)
```

## Key Components

### Cube Module (`navllm.cube`)

Implements the formal model from Section 3:

- **CubeSchema** (Definition 1): `S = <D, M>` with dimensions and measures
- **CubeView** (Definition 3): `V = <L_vec, φ, A>` with levels, filters, aggregations
- **NavigationAction** (Definition 4): Drill-down, roll-up, slice, dice operations
- **CubeEngine**: Executes OLAP queries and computes I_data

### LLM Module (`navllm.llm`)

Implements LLM integration from Section 5.3 and 6.2:

- **LLMClient**: Abstract interface for LLM APIs
- **PreferenceScorer**: Computes I_pref(V | Hist_t) via LLM
- **ExplanationGenerator**: Generates natural language rationales

### Navigation Module (`navllm.nav`)

Implements session management and recommendation:

- **Session** (Definition 5): `Σ = <s_0, ..., s_T>` with states
- **NavLLMRecommender** (Algorithm 1): Main recommendation engine

### Evaluation Module (`navllm.eval`)

Implements metrics from Section 7.1.4:

- Hit Rate@B
- Steps to First Hit
- Cumulative Interestingness
- Redundancy
- Coverage

## Quick Start

```python
from navllm.cube import CubeSchema, CubeView, PandasCubeEngine
from navllm.llm import OpenAIClient
from navllm.nav import Session, NavLLMRecommender, RecommenderConfig

# 1. Define schema
schema = CubeSchema(...)

# 2. Create engine
engine = PandasCubeEngine(schema, fact_df, dim_dfs)

# 3. Initialize LLM client
llm_client = OpenAIClient(api_key="...")

# 4. Create recommender
config = RecommenderConfig(lambda_data=0.4, lambda_pref=0.4, lambda_div=0.2)
recommender = NavLLMRecommender(schema, engine, llm_client, config)

# 5. Start session
session = Session()
initial_view = CubeView.create_overview(schema)
session.add_state(initial_view, "Show me the data")

# 6. Get recommendations
recommendations = recommender.recommend(session, "Where did sales drop?", "revenue")

# 7. Navigate
selected = recommendations[0]
new_view = selected.view
session.add_state(new_view, "Where did sales drop?")
```

## Installation

```bash
pip install -e ".[all]"
```

## Running Tests

```bash
pytest tests/
```

## License

MIT License
