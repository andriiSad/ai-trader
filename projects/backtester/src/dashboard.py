import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import yaml

from src.charts import build_fold_metrics_table, build_pair_metrics_table
from src.pipeline import run_backtest

st.set_page_config(page_title="Backtester Dashboard", layout="wide")


@st.cache_data
def load_results(results_path: str):
    with open(results_path) as f:
        return json.load(f)


@st.cache_data
def run_backtest_cached(config_json: str):
    config = json.loads(config_json)
    return run_backtest(config)


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
    long_only = st.sidebar.checkbox("Long Only", value=config["strategy"]["long_only"])

    config["strategy"]["threshold"] = threshold
    config["strategy"]["long_only"] = long_only

    config_json = json.dumps(config, sort_keys=True)
    results = run_backtest_cached(config_json)

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
    pair_metrics = pair_data["metrics"]

    st.subheader(f"Key Metrics — {pair}")
    pc1, pc2, pc3, pc4, pc5, pc6 = st.columns(6)
    pc1.metric("Sharpe", f"{pair_metrics['sharpe']:.2f}")
    pc2.metric("Max DD", f"{pair_metrics['max_drawdown'] * 100:.1f}%")
    pc3.metric("Win Rate", f"{pair_metrics['win_rate'] * 100:.1f}%")
    pc4.metric("Profit Factor", f"{pair_metrics['profit_factor']:.2f}")
    pc5.metric("Total Return", f"{pair_metrics['total_return'] * 100:.1f}%")
    pc6.metric("Trades", pair_metrics["num_trades"])

    st.header("Per-Fold Results")
    fold_df = build_fold_metrics_table(pair_data["per_fold"])
    st.dataframe(fold_df)

    st.header("Per-Pair Comparison")
    pair_df = build_pair_metrics_table({p: overall[p] for p in overall})
    st.dataframe(pair_df)

    st.header("Benchmark Comparison")
    bench = pair_data["benchmark"]
    bcol1, bcol2, bcol3 = st.columns(3)
    bcol1.metric("Strategy Return", f"{pair_metrics['total_return'] * 100:.1f}%")
    bcol2.metric("Buy & Hold Return", f"{bench['total_return'] * 100:.1f}%")
    outperformance = pair_metrics["total_return"] - bench["total_return"]
    bcol3.metric(
        "Outperformance",
        f"{outperformance * 100:.1f}%",
        delta=f"{outperformance * 100:.1f}%",
    )


if __name__ == "__main__":
    main()
