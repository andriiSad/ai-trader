"""Plotly chart generators for training results visualization."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


def accuracy_bar_chart(per_fold: list[dict]) -> go.Figure:
    """Bar chart of accuracy per fold with 0.5 reference line."""
    fold_ids = [f"Fold {f['fold_id']}" for f in per_fold]
    accuracies = [f["accuracy"] for f in per_fold]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=fold_ids,
            y=accuracies,
            marker_color="#5c6bc0",
            name="Accuracy",
            text=[f"{a:.3f}" for a in accuracies],
            textposition="outside",
        )
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="#ef5350", annotation_text="Baseline (0.5)")
    fig.update_layout(
        title="Per-Fold Accuracy",
        xaxis_title="Fold",
        yaxis_title="Accuracy",
        yaxis_range=[0, 1],
        height=400,
    )
    return fig


def feature_importance_chart(feature_importance: list[dict], top_n: int = 20) -> go.Figure:
    """Horizontal bar chart of top N features by importance."""
    top = feature_importance[:top_n]
    features = [f["feature"] for f in top][::-1]
    importances = [f["importance"] for f in top][::-1]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=features,
            x=importances,
            orientation="h",
            marker_color="#26a69a",
            name="Importance",
        )
    )
    fig.update_layout(
        title=f"Top {top_n} Feature Importance (avg gain)",
        xaxis_title="Importance (gain)",
        yaxis_title="Feature",
        height=max(400, top_n * 25),
    )
    return fig


def confusion_matrix_chart(confusion_matrix: list, fold_id: int) -> go.Figure:
    """Heatmap of confusion matrix for a fold."""
    cm = np.array(confusion_matrix)
    labels = ["DOWN", "UP"]

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale="Blues",
            text=cm,
            texttemplate="%{text}",
            showscale=False,
        )
    )
    fig.update_layout(
        title=f"Confusion Matrix – Fold {fold_id}",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=400,
        width=450,
    )
    return fig


def equity_curve_chart(per_fold: list[dict]) -> go.Figure:
    """Cumulative returns equity curve. Assume +1 for correct prediction, -1 for wrong."""
    fold_ids = []
    cumulative = []
    running = 0.0

    for fold in per_fold:
        accuracy = fold["accuracy"]
        test_samples = fold["test_samples"]
        correct = int(round(accuracy * test_samples))
        wrong = test_samples - correct
        pnl = correct * 1.0 - wrong * 1.0
        running += pnl
        fold_ids.append(f"Fold {fold['fold_id']}")
        cumulative.append(running)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fold_ids,
            y=cumulative,
            mode="lines+markers",
            line=dict(color="#5c6bc0", width=2),
            marker=dict(size=8),
            name="Cumulative PnL",
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#999")
    fig.update_layout(
        title="Equity Curve (Cumulative PnL)",
        xaxis_title="Fold",
        yaxis_title="Cumulative PnL (units)",
        height=400,
    )
    return fig


def pair_accuracy_chart(per_pair: dict) -> go.Figure:
    """Grouped bar chart comparing per-pair accuracy across folds."""
    if not per_pair:
        fig = go.Figure()
        fig.update_layout(title="Per-Pair Accuracy (no pair data)", height=400)
        return fig

    pairs = sorted(per_pair.keys())
    metrics = ["accuracy", "precision", "recall", "f1"]

    fig = go.Figure()
    colors = ["#5c6bc0", "#26a69a", "#ef5350", "#ffa726"]
    for i, metric in enumerate(metrics):
        values = [per_pair[p].get(metric, 0) for p in pairs]
        fig.add_trace(
            go.Bar(
                x=pairs,
                y=values,
                name=metric.capitalize(),
                marker_color=colors[i % len(colors)],
                text=[f"{v:.3f}" for v in values],
                textposition="outside",
            )
        )

    fig.update_layout(
        title="Per-Pair Accuracy Comparison",
        xaxis_title="Pair",
        yaxis_title="Score",
        yaxis_range=[0, 1],
        barmode="group",
        height=450,
    )
    return fig


def fold_timeline_chart(per_fold: list[dict]) -> go.Figure:
    """Timeline showing each fold's train/test date ranges as horizontal bars."""
    fig = go.Figure()

    for fold in per_fold:
        fold_label = f"Fold {fold['fold_id']}"
        fig.add_trace(
            go.Bar(
                y=[fold_label],
                x=[1],
                base=[0],
                orientation="h",
                marker_color="#5c6bc0",
                name=fold_label,
                showlegend=False,
                hovertext=(
                    f"Train: {fold['train_start']} → {fold['train_end']}<br>"
                    f"Test: {fold['test_start']} → {fold['test_end']}<br>"
                    f"Samples: {fold['train_samples']} train / {fold['test_samples']} test"
                ),
                hoverinfo="text",
            )
        )

    fig.update_layout(
        title="Walk-Forward Fold Timeline",
        xaxis_title="",
        yaxis_title="Fold",
        height=max(300, len(per_fold) * 50),
        xaxis=dict(showticklabels=False),
    )
    return fig


def build_results_dashboard(results_path: str, output_path: str) -> None:
    """Read evaluation_results.json, combine all charts into single HTML file."""
    with open(results_path) as f:
        results = json.load(f)

    html_parts: list[str] = []

    overall = results.get("overall_accuracy")
    if overall is not None:
        html_parts.append(
            f"<h2>Overall Accuracy: {overall:.3f}</h2>"
            f"<p>Precision: {results.get('overall_precision', 0):.3f} | "
            f"Recall: {results.get('overall_recall', 0):.3f} | "
            f"F1: {results.get('overall_f1', 0):.3f}</p>"
        )

    per_fold = results.get("per_fold", [])
    if per_fold:
        figs = [
            accuracy_bar_chart(per_fold),
            equity_curve_chart(per_fold),
            fold_timeline_chart(per_fold),
        ]
        for fig in figs:
            html_parts.append(fig.to_html(full_html=False, include_plotlyjs=False))

        for fold in per_fold:
            cm = fold.get("confusion_matrix")
            if cm:
                fig = confusion_matrix_chart(cm, fold["fold_id"])
                html_parts.append(fig.to_html(full_html=False, include_plotlyjs=False))

    feature_importance = results.get("feature_importance", [])
    if feature_importance:
        fig = feature_importance_chart(feature_importance)
        html_parts.append(fig.to_html(full_html=False, include_plotlyjs=False))

    per_pair = results.get("per_pair", {})
    if per_pair:
        fig = pair_accuracy_chart(per_pair)
        html_parts.append(fig.to_html(full_html=False, include_plotlyjs=False))

    plotly_cdn = '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'
    full_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Training Results Dashboard</title>
{plotly_cdn}
<style>body{{font-family:sans-serif;margin:20px;}}</style>
</head><body>
<h1 style="border-bottom:2px solid #333;padding-bottom:8px;">Training Results Dashboard</h1>
{"".join(html_parts)}
</body></html>"""

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(full_html, encoding="utf-8")
