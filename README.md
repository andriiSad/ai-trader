# AI Trader

Monorepo for crypto ML trading research projects.

## Projects

| Project | Description | Status |
|---------|-------------|--------|
| [scraper](projects/scraper/) | WhiteBIT candle data collector (OHLCV) via WebSocket | ✅ Complete |
| linear-regression | Linear regression baseline model | 🔜 Planned |

## Structure

```
ai-trader/
├── projects/
│   ├── scraper/          # WhiteBIT data collector
│   │   ├── scraper.py
│   │   ├── config.yaml
│   │   ├── requirements.txt
│   │   ├── tests/
│   │   ├── docs/
│   │   └── README.md
│   └── linear-regression/  # (planned)
├── shared/               # Shared utilities (future)
├── .gitignore
└── README.md
```

## Getting Started

Each project has its own `requirements.txt` and `README.md`. See project folders for details.

```bash
# Example: run the scraper
cd projects/scraper
pip install -r requirements.txt
python3 scraper.py backfill
```
