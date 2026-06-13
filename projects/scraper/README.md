# WhiteBIT Candle Data Collector

CLI tool to collect OHLCV candle data from WhiteBIT exchange via WebSocket API for ML model training.

## Features

- **Backfill**: fetch historical candle data with automatic resume
- **Live**: stream real-time candle updates with auto-reconnect
- **Configurable**: pairs, intervals, start date via YAML + CLI overrides
- **Resilient**: keepalive pings, graceful shutdown, gap detection

## Quick Start

```bash
pip install -r requirements.txt

# Backfill last 30 days (default: BTC_USDT, ETH_USDT × 1m/5m/15m/1h)
python3 scraper.py backfill

# Live streaming (Ctrl+C to stop)
python3 scraper.py live
```

## Configuration

Edit `config.yaml`:

```yaml
pairs:
  - BTC_USDT
  - ETH_USDT

intervals:
  - 1m
  - 5m
  - 15m
  - 1h

output_dir: data
start_date: null  # null = 30 days ago
backfill_chunk_size: 500
backfill_delay: 0.5
```

CLI overrides:

```bash
python3 scraper.py backfill --pairs BTC_USDT --intervals 1m --start-date 2026-01-01
python3 scraper.py live --output-dir /path/to/data
```

## Output

CSV files at `data/{PAIR}/{interval}.csv`:

```
data/
├── BTC_USDT/
│   ├── 1m.csv
│   ├── 5m.csv
│   ├── 15m.csv
│   └── 1h.csv
└── ETH_USDT/
    ├── 1m.csv
    └── ...
```

Columns: `timestamp,open,high,low,close,volume,deal` (standard OHLCV)

## Tests

```bash
python3 -m pytest tests/ -v
```

84 tests across: utils, backfill, live, resilience.

## Architecture

| Component | Description |
|-----------|-------------|
| `Backfiller` | Historical data fetch via `candles_request` with chunking and resume |
| `LiveCollector` | Real-time streaming via `candles_subscribe`, one WS per interval |
| CSV utilities | File creation, append, update-or-append, timestamp tracking |

## WhiteBIT WebSocket API

- Endpoint: `wss://api.whitebit.com/ws`
- No authentication required for public candle channels
- Keepalive: ping every 50s (server closes at 60s)
- Rate limit: 200 requests/minute for `candles_request`
