import json

import streamlit as st
import yaml

from src.benchmark import buy_and_hold_returns  # noqa: F401
from src.charts import (
    build_drawdown_chart,  # noqa: F401
    build_equity_curve_chart,  # noqa: F401
    build_fold_metrics_table,
    build_pair_metrics_table,
    build_trade_markers_chart,  # noqa: F401
)
from src.data_loader import load_all_candles, load_predictions  # noqa: F401
from src.pipeline import run_backtest

st.set_page_config(page_title="Backtester Dashboard", layout="wide")


@st.cache_data
def load_results(results_path: str):
    with open(results_path) as f:
        return json.load(f)


def main():
    st.title("Backtester Dashboard")

    st.sidebar.header("Configuration")
    config_path = st.sidebar.text_input("Config path", "config.yaml")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    for key, value in config.items():
        if isinstance(value, (str, int, float, bool)):
            st.sidebar.write(f"**{key}:** {value}")

    st.sidebar.header("Strategy Parameters")
    threshold = st.sidebar.slider(
        "Probability Threshold", 0.0, 1.0, config["strategy"]["threshold"], 0.01
    )
    st.sidebar.checkbox("Long Only", value=config["strategy"]["long_only"])

    results_path = config["output"]["results_path"]
    try:
        results = load_results(results_path)
    except FileNotFoundError:
        st.info(
            "No saved results found. Run `python backtester.py run` first, or click 'Run Backtest' below."
        )
        if st.button("Run Backtest"):
            results = run_backtest(config)
            st.rerun()
        return

    overall = results["overall"]
    first_pair = list(overall.keys())[0]
    metrics = overall[first_pair]

    st.header("Key Metrics")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Sharpe", f"{metrics['sharpe']:.2f}")
    col2.metric("Max DD", f"{metrics['max_drawdown'] * 100:.1f}%")
    col3.metric("Win Rate", f"{metrics['win_rate'] * 100:.1f}%")
    col4.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
    col5.metric("Total Return", f"{metrics['total_return'] * 100:.1f}%")
    col6.metric("Trades", metrics["num_trades"])

    pair = st.selectbox("Select Pair", list(overall.keys()))
    pair_data = results["per_pair"][pair]

    st.header("Per-Fold Results")
    fold_df = build_fold_metrics_table(pair_data["per_fold"])
    st.dataframe(fold_df)

    st.header("Per-Pair Comparison")
    pair_df = build_pair_metrics_table({p: overall[p] for p in overall})
    st.dataframe(pair_df)

    st.header("Benchmark Comparison")
    bench = pair_data["benchmark"]
    bcol1, bcol2, bcol3 = st.columns(3)
    bcol1.metric("Strategy Return", f"{metrics['total_return'] * 100:.1f}%")
    bcol2.metric("Buy & Hold Return", f"{bench['total_return'] * 100:.1f}%")
    outperformance = metrics["total_return"] - bench["total_return"]
    bcol3.metric(
        "Outperformance",
        f"{outperformance * 100:.1f}%",
        delta=f"{outperformance * 100:.1f}%",
    )

    st.header("Threshold Tuning")
    st.write("Adjust the threshold slider in the sidebar to see how it affects performance.")
    if threshold != config["strategy"]["threshold"]:
        st.warning(f"Threshold changed to {threshold}. Re-run backtest to see updated results.")


if __name__ == "__main__":
    main()
