# NavLLM

**Conversational OLAP navigation powered by LLMs - combining data interestingness with user preferences**

NavLLM is a system for intelligent, conversational navigation over multidimensional data cubes (OLAP). It combines the analytical power of traditional OLAP engines with large language models (LLMs) to provide personalized, context-aware view recommendations during exploratory data analysis.

## 🎯 Key Features

- **Conversational Interface**: Navigate data cubes using natural language queries
- **Hybrid Recommendation System**: Combines three scoring components:
  - **Data-driven interestingness**: Statistical deviation detection and anomaly scoring
  - **Preference-based relevance**: LLM-powered understanding of user intent from conversation history
  - **Diversity-aware selection**: Graph-based scoring to avoid redundant views
- **Verifiable Computations**: All numerical analysis performed by conventional OLAP engine, not the LLM
- **Explainable Recommendations**: Natural language explanations for each suggested view
- **Flexible LLM Backend**: Supports OpenAI GPT-4, Google Gemini, and DeepSeek

## 🏗️ Architecture

```
┌─────────────┐         ┌──────────────────────┐         ┌─────────────────┐
│             │         │  LLM Navigation      │         │                 │
│  Analyst    │◄────────│      Engine          │◄────────│  Cube Engine    │
│             │         │                      │         │   (PostgreSQL)  │
└─────────────┘         │  • Candidate Gen     │         │                 │
                        │  • Preference Scorer │         │  • Star Schema  │
   Convers. Interface   │  • Diversity Scorer  │         │  • View Mater.  │
                        │  • Utility Ranker    │         │  • Stats Calc.  │
                        │  • Explan. Generator │         │                 │
                        └──────────────────────┘         └─────────────────┘
                                  │
                                  ▼
                        ┌──────────────────┐
                        │  External LLM API│
                        │  (DeepSeek/GPT)  │
                        └──────────────────┘
```

## 📦 Installation

### Prerequisites
- Python 3.10+
- PostgreSQL 15+
- Git

### Setup

1. **Clone the repository:**
   ```bash
   git clone git@github.com:xiufengliu/NAVLLM.git
   cd NAVLLM
   ```

2. **Install dependencies:**
   ```bash
   pip install -e .
   ```

3. **Set up PostgreSQL:**
   ```bash
   # Create database
   createdb navllm_db
   
   # Run schema setup (if provided)
   psql navllm_db < configs/schema.sql
   ```

4. **Configure LLM API:**
   Create a `.env` file with your API key:
   ```bash
   echo "DEEPSEEK_API_KEY=your_api_key_here" > .env
   # Or use OpenAI/Gemini:
   # echo "OPENAI_API_KEY=your_api_key_here" > .env
   ```

## 🚀 Quick Start

### Example Session

```python
from navllm import NavLLMEngine
from navllm.cube import CubeEngine

# Initialize engines
cube_engine = CubeEngine(db_uri="postgresql://localhost/navllm_db")
nav_engine = NavLLMEngine(
    cube_engine=cube_engine,
    llm_backend="deepseek",  # or "openai", "gemini"
    weights={
        "data": 0.4,
        "pref": 0.4,
        "div": 0.2
    }
)

# Start from overview
current_view = cube_engine.get_overview("SalesCube")

# User utterance
utterance = "Show me where sales dropped in 2023"

# Get recommendations
recommendations = nav_engine.recommend_next_views(
    current_view=current_view,
    utterance=utterance,
    top_k=5
)

# Display recommendations with explanations
for i, rec in enumerate(recommendations, 1):
    print(f"\n{i}. {rec['view_description']}")
    print(f"   Utility Score: {rec['utility']:.2f}")
    print(f"   Explanation: {rec['explanation']}")
```

### Running Experiments

```bash
# Download sample datasets
python scripts/download_data.py

# Preprocess data
python scripts/preprocess_data.py --cube SalesCube

# Run example session
python scripts/example_session.py --cube SalesCube --task sales_decline

# Run full experiments
python scripts/run_experiments.py --all
```

## 📊 Supported Cubes

NavLLM comes with three pre-configured multidimensional cubes:

1. **SalesCube** (Retail)
   - Dimensions: Time (day→year), Product (item→category), Store (store→state)
   - Measures: units_sold, revenue
   - Source: M5 Forecasting dataset (Walmart sales)

2. **ManufacturingCube** (Operations)
   - Dimensions: Time (shift→month), Line (machine→plant), Product (variant→category)
   - Measures: throughput, defect_count, defect_rate

3. **AirQualityCube** (Environmental)
   - Dimensions: Time (hour→season), Location (station→region), Pollutant
   - Measures: concentration, air_quality_index

## 🧪 Testing

```bash
# Run unit tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_cube.py -v

# Run with coverage
python -m pytest --cov=src/navllm tests/
```

## 📁 Project Structure

```
NAVLLM/
├── src/navllm/          # Core library
│   ├── cube/            # OLAP cube engine
│   ├── llm/             # LLM integration
│   ├── api/             # API endpoints
│   └── eval/            # Evaluation metrics
├── scripts/             # Experiment and data scripts
├── configs/             # Configuration files
├── data/                # Sample datasets
├── tests/               # Unit tests
└── agents/              # Agent implementations
```

## ⚙️ Configuration

Edit `configs/cubes.py` to add custom cubes or modify existing ones:

```python
CUBES = {
    "YourCube": {
        "dimensions": [
            {"name": "Time", "levels": ["day", "month", "year"]},
            {"name": "Region", "levels": ["city", "state", "country"]},
        ],
        "measures": ["sales", "profit"],
        "db_table": "your_fact_table"
    }
}
```

Adjust utility function weights in your engine initialization:

```python
nav_engine = NavLLMEngine(
    cube_engine=cube_engine,
    weights={
        "data": 0.5,    # Increase for more data-driven recommendations
        "pref": 0.3,    # Increase for more preference alignment
        "div": 0.2      # Increase for more diverse exploration
    }
)
```

## 📈 Performance

On benchmark cubes with 24 participants:
- **24% higher cumulative interestingness** vs. LLM-only baselines
- **56% reduction in redundant exploration** vs. purely data-driven heuristics
- **38% higher usefulness ratings** (5.8/7 vs. 4.2/7) compared to manual navigation

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

Xiufeng Liu - xiuli@dtu.dk

Project Link: [https://github.com/xiufengliu/NAVLLM](https://github.com/xiufengliu/NAVLLM)

## 🙏 Acknowledgments

- M5 Forecasting Competition for the retail sales dataset
- PostgreSQL and FastAPI communities
- LLM API providers (DeepSeek, OpenAI, Google)
