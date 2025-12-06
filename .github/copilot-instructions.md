# Copilot Instructions

## Architecture Snapshot
- `app.py` is the primary Streamlit UI; it orchestrates credential gating, pulls DB data via `src/querys.py`, reads GeoJSON/shapefiles from `data/shapefiles/`, and loads pickled scikit-learn models from `models/models_trained/`.
- `main_api.py` exposes a lightweight Flask service that lazily loads the same pickled models (metro + metrobus classifiers) and serves crime/affluence predictions; keep it dependency-light because it is deployed independently of the Streamlit UI.
- Supporting modules live in `src/`: `colors.py` centralizes palette dictionaries, `plots.py` contains reusable Plotly chart builders, and `querys.py` houses all SQL Server accessors (pyodbc + pandas) used by both `app.py` and notebooks.
- Data artifacts are intentionally committed under `data/` (CSV cubes, shapefiles, SARIMA outputs) and `assets/images/` (icons, GeoJSON). Always reference them via the relative paths already used in `app.py` to avoid breaking the UI.

## Environment & Workflows
- Use `uv venv` (already created as `.venv`) and `uv pip install -r requirements.txt` for dependency sync; reference that when documenting setup steps.
- The app expects environment variables `SERVER`, `DATABASE`, `USERNAME`, `PASSWORD` (plus `PASSWORD` reused for Streamlit auth) loaded via `python-dotenv`. Never hardcode secrets; load via `.env` in development and production env vars otherwise.
- SQL connections rely on "ODBC Driver 17 for SQL Server"; remind contributors to install it locally or in Docker before running data queries.
- Run the Streamlit UI with `streamlit run app.py`; run the Flask API with `python main_api.py`. There is no automated test suite—manual verification through these entrypoints is the norm.

## Coding Conventions & Patterns
- Keep new business logic inside `src/` modules and import from `app.py` rather than expanding the already-large Streamlit script; follow the existing relative import style (`from src.plots import ...`).
- `src/querys.py` opens a new pyodbc connection per helper; match this pattern and use existing filters (transport, level_div, etc.) instead of inlining SQL in the UI.
- Chart additions should follow `src/plots.py` patterns: accept pandas DataFrames, derive labels/colours via `src/colors.py`, and return Plotly figs compatible with `plotly_events`.
- Geospatial assets assume EPSG:32614 (UTM 14N) shapefiles converted in Streamlit; when adding new layers, process them offline into `data/shapefiles/` and load with geopandas as seen in `app.py`.
- Models are serialized with pickle; if retraining, drop artefacts in `models/models_trained/{final,test}/` and keep filenames aligned with the format used in both entrypoints (`clf_crime_{system}_dataset_{group}_wm_2_mas_perc.pkl`).
- Follow repo-wide editing constraints: ASCII only unless necessary, keep comments minimal but meaningful, and avoid reverting user-made changes.

You are assisting in a Python project with the following structure and constraints.
Follow all instructions exactly.

Architecture Overview

The main Streamlit interface lives in app.py.
It handles:

credential gating (env-loaded password),

DB reads via src/querys.py,

loading GeoJSON/shapefiles from data/shapefiles/,

loading scikit-learn pickled models from models/models_trained/.

A lightweight Flask API exists in main_api.py.
It lazily loads the same crime/affluence classifiers and exposes prediction endpoints.
Keep it dependency-light, since this service deploys independently.

Support modules under src/:

colors.py: centralized palette dicts.

plots.py: Plotly figure builders, reusable across the app.

querys.py: SQL Server access helpers via pyodbc + pandas.

Data assets:

data/ holds CSVs, SARIMA outputs, shapefiles.

assets/images/ holds icons and GeoJSON.
Always reference paths exactly as used in app.py.

Environment Requirements

Virtual env is created with uv:

uv venv
uv pip install -r requirements.txt


Environment variables required:
SERVER, DATABASE, USERNAME, PASSWORD
Load them with python-dotenv via .env in development.

Never hardcode credentials.
Always use os.getenv() or the existing env loader.

SQL connections rely on ODBC Driver 17 for SQL Server.
Before writing code that queries data, ensure the developer is reminded to install this driver.

Entrypoints:

Streamlit: streamlit run app.py

Flask API: python main_api.py

Coding Guidelines

Place new business logic inside src/ instead of expanding app.py.

Follow existing import style:

from src.plots import build_plot_x
from src.querys import fetch_data_y


In src/querys.py:

Each helper opens its own pyodbc connection.

Follow existing query patterns and filters (transport, level_div, etc.).

Never inline SQL inside Streamlit.

In src/plots.py:

Accept pandas DataFrames.

Use colors from src/colors.py.

Return Plotly figures compatible with plotly_events.

Geospatial handling:

Shapefiles are preprocessed to EPSG:32614.

Load with geopandas exactly as in app.py.

New layers must be preprocessed offline and saved to data/shapefiles/.

Model handling:

Models are pickled and stored in models/models_trained/{final,test}/.

Filename pattern must remain:
clf_crime_{system}_dataset_{group}_wm_2_mas_perc.pkl

Keep compatibility with both app.py and main_api.py.

Repo-wide conventions:

ASCII unless necessary.

Minimal but meaningful comments.

Do not revert developer changes.

Maintain relative paths exactly.

Copilot Tasks

When writing or editing code, Copilot must:

Respect module boundaries (UI logic in app.py, business logic in src/).

Use existing helpers in src/querys.py rather than opening new DB connections elsewhere.

Generate figures following the structure of src/plots.py.

Load models and data via the exact relative paths already in the repo.

Avoid introducing heavyweight dependencies to the Flask API.

Produce code compatible with both local development and deployment without modification.