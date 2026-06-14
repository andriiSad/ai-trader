# Backtester

Backtesting engine for crypto ML trading strategies.

## Quick Start

```bash
pip install -r requirements.txt
```

## Commands

### Run Backtest

```bash
python backtester.py run
python backtester.py run --config path/to/config.yaml
```

### Launch Dashboard

```bash
python backtester.py dashboard
```

## Configuration

All settings live in `config.yaml`:

| Field | Description |
|-------|-------------|
| `pairs` | Trading pairs (e.g. `BTC_USDT`, `ETH_USDT`) |
| `interval` | Candle interval (e.g. `5m`, `1h`) |
| `predictions_path` | Path to trainer predictions CSV |
| `candles_dir` | Directory with candle data |
| `strategy.threshold` | Probability threshold for signal generation |
| `strategy.long_only` | Only take long positions |
| `costs.fee_pct` | Trading fee percentage |
| `costs.slippage_pct` | Slippage percentage |
| `position.risk_per_trade` | Fraction of capital risked per trade |
| `position.max_duration_candles` | Max hold duration in candles |
| `output.results_path` | Path for backtest results JSON |
| `output.tear_sheet_path` | Path for tear sheet HTML |

## Dependencies

- streamlit
- vectorbt
- quantstats
- plotly
- pandas
- numpy
- pyyaml
- pytest
