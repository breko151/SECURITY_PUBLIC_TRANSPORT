# Security Public Transport

Project to predict crime risk in public transport in CDMX.

## Setup

1. Install uv: `pip install uv`
2. Create venv: `uv venv`
3. Install dependencies: `uv pip install -e .`

## Training Models

To retrain models:

```bash
python src/training/crime_model_train.py --system metro --group 3
```

## Architecture

- `app.py`: Streamlit UI
- `main_api.py`: Flask API
- `src/`: Source code
- `data/`: Data files
- `models/`: Trained models
