import json
from pathlib import Path

import pandas as pd

from src.benchmark import buy_and_hold_returns, compute_benchmark_metrics
from src.data_loader import load_all_candles, load_predictions
from src.engine import BacktestEngine
from src.metrics import compute_all_metrics
from src.position import PositionManager
from src.strategy import ThresholdStrategy


def run_backtest(config: dict) -> dict:
    predictions = load_predictions(config["predictions_path"], config.get("pairs"))

    all_candles = load_all_candles(config["candles_dir"], config["pairs"], config["interval"])

    strategy = ThresholdStrategy(
        threshold=config["strategy"]["threshold"], long_only=config["strategy"]["long_only"]
    )
    position_manager = PositionManager(
        risk_per_trade=config["position"]["risk_per_trade"],
        max_duration_candles=config["position"]["max_duration_candles"],
        fee_pct=config["costs"]["fee_pct"],
        slippage_pct=config["costs"]["slippage_pct"],
    )
    initial_capital = config.get("initial_capital", 10000)

    engine = BacktestEngine(
        strategy=strategy,
        position_manager=position_manager,
        initial_capital=initial_capital,
        fee_pct=config["costs"]["fee_pct"],
        slippage_pct=config["costs"]["slippage_pct"],
    )

    all_results = {}
    all_metrics = {}

    for pair in config["pairs"]:
        pair_preds = predictions[predictions["pair"] == pair].copy()
        pair_candles = all_candles[pair]

        fold_results = engine.run_walk_forward(pair_preds, pair_candles)

        per_fold = []
        for i, result in enumerate(fold_results):
            fold_initial = initial_capital if i == 0 else fold_results[i - 1].final_value
            metrics = compute_all_metrics(result.equity_curve, result.trade_log, fold_initial)
            metrics["fold_id"] = result.fold_id
            metrics["fold_start"] = str(result.fold_start)
            metrics["fold_end"] = str(result.fold_end)
            per_fold.append(metrics)

        chained_curves = []
        cumulative_offset = 0.0
        for i, result in enumerate(fold_results):
            if i > 0:
                cumulative_offset += result.equity_curve.iloc[0] - initial_capital
            chained_curves.append(result.equity_curve - cumulative_offset)
        combined_equity = pd.concat(chained_curves)
        combined_trades = [t for r in fold_results for t in r.trade_log]
        combined_metrics = compute_all_metrics(combined_equity, combined_trades, initial_capital)

        benchmark_equity = buy_and_hold_returns(pair_candles, initial_capital)
        benchmark_metrics = compute_benchmark_metrics(pair_candles, initial_capital)

        all_results[pair] = {
            "per_fold": per_fold,
            "metrics": combined_metrics,
            "benchmark": benchmark_metrics,
            "equity_curve": combined_equity,
            "benchmark_curve": benchmark_equity,
            "trade_log": combined_trades,
        }
        all_metrics[pair] = combined_metrics

    results_path = Path(config["output"]["results_path"])
    results_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "overall": {pair: m for pair, m in all_metrics.items()},
        "per_pair": {
            pair: {
                "per_fold": results["per_fold"],
                "metrics": results["metrics"],
                "benchmark": results["benchmark"],
                "trade_count": len(results["trade_log"]),
            }
            for pair, results in all_results.items()
        },
        "config": config,
    }

    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    try:
        import quantstats as qs

        tear_sheet_path = config["output"].get("tear_sheet_path")
        if tear_sheet_path:
            for pair, results in all_results.items():
                equity = results["equity_curve"]
                equity.index = pd.to_datetime(equity.index)
                qs.reports.html(
                    equity,
                    output=f"{tear_sheet_path.replace('.html', f'_{pair}.html')}",
                    title=f"Backtest: {pair}",
                )
    except ImportError:
        pass

    return output
