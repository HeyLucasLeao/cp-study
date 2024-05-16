import numpy as np
from typing import List
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.calibration import calibration_curve


def efficiency_curve(clf, X: np.ndarray, fig_type=None):

    def get_error_metrics(clf, X: np.ndarray) -> List:
        error_rate = {
            k: {} for k in [0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05]
        }
        for error in error_rate:
            predict_set = clf.predict_set(X, alpha=error)
            error_rate[error]["efficiency"] = np.sum(
                [np.sum(p) == 1 for p in predict_set]
            ) / len(predict_set)
            error_rate[error]["validity"] = np.sum(predict_set) / len(predict_set)
        return error_rate

    error_rate = get_error_metrics(clf, X)

    df = pd.DataFrame(error_rate).T

    # Create the bar chart
    fig = px.line(
        df,
        x=df.index,
        y=["efficiency", "validity"],
        labels={"value": "Score"},
        markers=True,
        title="Efficiency & Validity Curve",
        color_discrete_sequence=["darkblue", "orange"],
        width=800,
        height=400,
    )
    fig.update_layout(hovermode="x")
    fig.update_traces(hovertemplate="%{y}")
    fig.update_layout(legend=dict(title="Metric"))
    fig.update_yaxes(title_text="Score")
    fig.update_xaxes(title_text="Error Rate")
    return fig.show(fig_type)


def reliability_curve(clf, X, y, n_bins=15, fig_type=None) -> go.Figure:

    y_prob = clf.predict_proba(X)[:, 1]

    v_prob_true, v_prob_pred = calibration_curve(
        y, y_prob, n_bins=n_bins, strategy="quantile"
    )

    fig = go.Figure()

    # Add traces for each model

    fig.add_trace(
        go.Scatter(
            x=v_prob_pred, y=v_prob_true, mode="lines+markers", name="RandomForest"
        )
    )

    # Add a trace for the perfectly calibrated line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfectly calibrated",
            line=dict(dash="dash", color="grey"),
        )
    )

    fig.update_layout(
        title="Reliability Curve",
        xaxis_title="Mean predicted probability",
        yaxis_title="Fraction of positives",
        legend_title="Model",
        autosize=False,
    )

    return fig.show(fig_type)


def histogram(clf, X, nbins=15, fig_type=None):
    y_prob = clf.predict_proba(X)[:, 1]
    fig = px.histogram(y_prob, nbins=nbins)
    fig.update_layout(
        title="Histogram of Predicted Scores",
        xaxis_title="Predicted Scores",
        yaxis_title="Count",
        legend_title="Modelos",
        autosize=False,
    )
    fig.update_layout(hovermode="x")
    fig.update_traces(hovertemplate="%{y}")
    fig.update_layout(showlegend=False)
    return fig.show(fig_type)
