.PHONY: test lint format install-scraper

# Run all tests
test:
	cd projects/scraper && python3 -m pytest tests/ -v --tb=short

# Run tests with coverage
test-cover:
	cd projects/scraper && python3 -m pytest tests/ -v --tb=short --cov=scraper --cov-report=term-missing

# Lint all code
lint:
	ruff check .

# Format all code
format:
	ruff format .

# Install scraper dependencies
install-scraper:
	cd projects/scraper && pip install -r requirements.txt

# Run scraper backfill
backfill:
	cd projects/scraper && python3 scraper.py backfill

# Run scraper live
live:
	cd projects/scraper && python3 scraper.py live
