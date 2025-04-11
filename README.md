# Autonomous Data Pipeline Builder

An intelligent Auto-EDA and preprocessing agent that autonomously analyzes datasets and builds ML-ready data pipelines.

## Features

- **Automated Data Ingestion**: Support for CSV, JSON, and SQL data sources
- **Intelligent EDA**: Autonomous exploration of data distributions, missing values, and correlations
- **Smart Preprocessing**: AI-powered feature engineering and preprocessing suggestions
- **Pipeline Generation**: Creates reusable scikit-learn compatible pipelines
- **Model Recommendations**: Suggests suitable algorithms based on data characteristics
- **Explainable Decisions**: Transparent reasoning for all pipeline choices
- **Interactive UI**: Gradio interface for easy interaction and pipeline customization

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/auto_pipeline_builder.git
cd auto_pipeline_builder

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from auto_pipeline.core import DataPipeline
from auto_pipeline.agents import EDAAgent, PreprocessingAgent

# Initialize the pipeline
pipeline = DataPipeline()

# Load your data
pipeline.load_data("path/to/your/data.csv")

# Run autonomous analysis
pipeline.analyze()

# Get preprocessing recommendations
recommendations = pipeline.get_recommendations()

# Generate and save the pipeline
pipeline.generate()
pipeline.save("my_pipeline.pkl")
```

## Project Structure

```
auto_pipeline_builder/
├── src/
│   └── auto_pipeline/
│       ├── core/        # Core pipeline functionality
│       ├── agents/      # AI agents for analysis
│       └── utils/       # Helper functions
├── notebooks/          # Example notebooks
├── tests/             # Unit tests
├── configs/           # Configuration files
└── examples/          # Example datasets
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 