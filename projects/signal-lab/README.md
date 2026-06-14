# Signal Lab

Signal generation and analysis for crypto trading.

## Quick Start

```bash
cd projects/signal-lab
pip install -r requirements.txt
python signal_lab.py download
```

This downloads BTC/USDT 4h candles from Binance and saves to `data/BTC_USDT/4h.csv`.

## Commands

```bash
python signal_lab.py download    # Download candle data from Binance
```

## Configuration

All settings are in `config.yaml`:

```yaml
pair: BTC_USDT        # Trading pair
interval: 4h          # Candle interval
start_date: null      # Start date (defaults to 2+ years ago)
data_dir: data        # Output directory for CSVs
```

## Project Structure

```
signal_lab.py          CLI entry point (argparse)
config.yaml            Pipeline configuration
src/
├── __init__.py
└── data.py            Binance download + CSV save
tests/
└── test_data.py
data/                  Downloaded CSVs (gitignored)
```

## Dependencies

- python-binance – Binance REST API client
- pandas – data processing
- numpy – numerical operations
- pyyaml – config parsing
