# Trainer

ML training pipeline for crypto trading with LightGBM walk-forward evaluation.

## Quick Start

```bash
cd projects/trainer
pip install -r requirements.txt
python trainer.py run
```

This executes the full pipeline: features → train → evaluate → visualize.

## Individual Commands

```bash
python trainer.py features      # Generate features + labels only
python trainer.py train          # Features + walk-forward training
python trainer.py evaluate       # Features + train + save results JSON
python trainer.py visualize      # Build HTML dashboards from results
```

## Data Exploration

Standalone data exploration dashboard (separate from training pipeline):

```bash
python explore.py --data-dir ../../data --pairs BTC_USDT ETH_USDT --output ../../data/reports/data_exploration.html
```

### Exploration Charts

- **Candlestick** – OHLC with volume overlay
- **Volume Profile** – volume distribution across price levels
- **Return Distribution** – 1-bar and 5-bar return histograms
- **Correlation Heatmap** – feature correlations (returns, volume change, range)
- **Feature Distributions** – box plots of derived features

## Configuration

All settings are in `config.yaml`:

```yaml
pairs:                     # Trading pairs to train on
  - BTC_USDT
  - ETH_USDT

interval: 5m               # Candle interval
data_dir: ../../data       # Root data directory

features:
  prediction_horizon: 5    # Bars ahead for label generation
  label_threshold: 0.0     # Minimum return for UP label

walk_forward:
  train_window: "3ME"      # Training window per fold
  test_window: "1ME"       # Test window per fold
  step: "1ME"              # Step size between folds

model:
  objective: binary
  metric: binary_logloss
  num_leaves: 31
  learning_rate: 0.05
  feature_fraction: 0.8
  bagging_fraction: 0.8
  bagging_freq: 5
  seed: 42

output:
  results_path: ../../data/reports/evaluation_results.json
```

## Architecture

```
trainer.py              CLI entry point (argparse)
explore.py              Standalone data exploration CLI
config.yaml             Pipeline configuration
src/
├── data_loader.py      Load candle CSV files
├── features.py         Technical indicators (RSI, MACD, Bollinger, SMA/EMA, etc.)
├── labels.py           Binary UP/DOWN label generation from forward returns
├── walk_forward.py     Time-series walk-forward split engine
├── trainer.py          LightGBM training + walk-forward evaluation
├── metrics.py          Accuracy, precision, recall, F1, confusion matrix
├── result_charts.py    Training results Plotly charts
├── explore_charts.py   Data exploration Plotly charts
└── pipeline.py         Full pipeline orchestrator
```

### Data Flow

```
CSV files → data_loader → features + labels → walk_forward splits
→ LightGBM train per fold → metrics → evaluation_results.json
→ result_charts → training_results.html
→ explore_charts → data_exploration.html
```

## Output

- `data/reports/evaluation_results.json` – structured evaluation results
- `data/reports/training_results.html` – interactive training dashboard
- `data/reports/data_exploration.html` – interactive data exploration dashboard

## Dependencies

- pandas, numpy – data processing
- lightgbm – gradient boosting model
- plotly – interactive charts
- pyyaml – config parsing
