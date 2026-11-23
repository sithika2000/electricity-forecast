from __future__ import annotations

# pip install streamlit plotly pandas numpy scikit-learn xgboost lightgbm catboost statsmodels joblib

from pathlib import Path
from typing import Optional
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="GridPulse LK — Ensemble Load Forecasting Dashboard",
    page_icon="⚡",
    layout="wide",
)

CLEAN_PATH = Path("data/processed/sl_electricity_load_clean.csv")
ARTIFACT_DIR = Path("artifacts/models")
BUNDLE_PATH = ARTIFACT_DIR / "ensemble_bundle.joblib"
METRICS_PATH = ARTIFACT_DIR / "model_metrics.csv"
PREDS_PATH = ARTIFACT_DIR / "test_predictions.csv"
IMP_PATH = ARTIFACT_DIR / "feature_importance.csv"


@st.cache_resource
def load_bundle():
    if not BUNDLE_PATH.exists():
        return None
    return joblib.load(BUNDLE_PATH)


@st.cache_data
def load_csv_safe(path: Path, parse_dates: bool = False):
    if not path.exists():
        return None
    if parse_dates:
        return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    return pd.read_csv(path)


def predict_with_model(
    model_key: str,
    models: dict,
    X_input: pd.DataFrame,
    y_history: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Run predictions cleanly across linear/tree/stacking/hybrid models."""
    if model_key == "Stacking_Ensemble":
        stack = models["stacking"]
        base_preds = [
            est.predict(X_input)
            for est in stack["base_models"].values()
        ]
        return np.asarray(stack["meta_model"].predict(np.column_stack(base_preds)), dtype=float)

    if model_key == "Hybrid_ARIMA_RF":
        arima_fit = models["hybrid_arima_fit"]
        rf_res = models["hybrid_rf_residual"]
        res_pred = np.asarray(rf_res.predict(X_input), dtype=float)

        # For single or small custom rows, use last ARIMA forecast step; for test set, apply series
        if len(X_input) == 1:
            arima_component = float(arima_fit.forecast(steps=1).iloc[0])
            return np.array([arima_component + res_pred[0]], dtype=float)

        if y_history is not None and len(y_history) == len(X_input):
            applied = arima_fit.apply(y_history)
            arima_component = np.asarray(applied.fittedvalues, dtype=float)
            if len(arima_component) != len(X_input):
                arima_component = np.asarray(arima_fit.forecast(steps=len(X_input)), dtype=float)
        else:
            arima_component = np.asarray(arima_fit.forecast(steps=len(X_input)), dtype=float)

        return arima_component + res_pred

    key_map = {
        "Baseline_LinearRegression": "baseline_lr",
        "Bagging_RandomForest": "random_forest",
        "Bagging_Regressor": "bagging_regressor",
        "Boosting_XGBoost": "xgboost",
        "Boosting_LightGBM": "lightgbm",
        "Boosting_CatBoost": "catboost",
    }
    estimator = models[key_map[model_key]]
    return np.asarray(estimator.predict(X_input), dtype=float)


def main():
    st.title("⚡ GridPulse LK — Sri Lankan Electricity Load Ensemble Dashboard")
    st.caption("Compare Bagging, Boosting, Stacking, and Hybrid forecasting models, inspect feature drivers, and test custom inputs.")

    bundle = load_bundle()
    clean_df = load_csv_safe(CLEAN_PATH, parse_dates=True)
    metrics_df = load_csv_safe(METRICS_PATH)
    preds_df = load_csv_safe(PREDS_PATH, parse_dates=True)
    imp_df = load_csv_safe(IMP_PATH)

    if bundle is None or clean_df is None:
        st.error(
            "Missing saved artifacts (`artifacts/models/ensemble_bundle.joblib`) or cleaned data "
            "(`data/processed/sl_electricity_load_clean.csv`). Run Steps 2, 4, and 6 first."
        )
        return

    models = bundle["models"]
    feature_cols = bundle["feature_cols"]
    target_col = bundle["target_col"]

    available_models = [
        m for m in [
            "Baseline_LinearRegression",
            "Bagging_RandomForest",
            "Bagging_Regressor",
            "Boosting_XGBoost",
            "Boosting_LightGBM",
            "Boosting_CatBoost",
            "Stacking_Ensemble",
            "Hybrid_ARIMA_RF",
        ]
        if (preds_df is not None and m in preds_df.columns) or m in ("Stacking_Ensemble", "Hybrid_ARIMA_RF")
    ]

    with st.sidebar:
        st.header("Controls")
        selected_model = st.selectbox("Select Forecasting Model", options=available_models, index=0)
        metric_choice = st.selectbox("Comparison Metric", options=["RMSE", "MAE", "MAPE_%", "R2", "Train_Time_s"], index=0)
        window_points = st.slider("Test Window Points to Plot", min_value=50, max_value=1000, value=200, step=50)

    # 1. Dataset Overview
    st.subheader("1. Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Clean Rows", f"{len(clean_df):,}")
    c2.metric("Feature Count", f"{len(feature_cols)}")
    c3.metric("Start Date", str(clean_df.index.min())[:10])
    c4.metric("End Date", str(clean_df.index.max())[:10])

    with st.expander("Preview Cleaned Dataset & Summary Stats"):
        st.dataframe(clean_df.head(10), width="stretch")
        st.dataframe(
            clean_df[[target_col] + [c for c in ["temperature_c", "humidity_pct", "rainfall_mm"] if c in clean_df.columns]]
            .describe()
            .round(2),
            width="stretch",
        )

    # 2. Model Comparison Table + Bar Chart
    st.subheader("2. Model Evaluation Table & Comparison Chart")
    if metrics_df is not None:
        m_display = metrics_df.copy()
        if m_display.columns[0] != "Model":
            m_display = m_display.rename(columns={m_display.columns[0]: "Model"})
        st.dataframe(m_display, width="stretch")

        if metric_choice in m_display.columns and "Model" in m_display.columns:
            fig_bar = px.bar(
                m_display.sort_values(metric_choice, ascending=(metric_choice != "R2")),
                x="Model",
                y=metric_choice,
                title=f"Model Comparison by {metric_choice}",
                text_auto=True,
            )
            st.plotly_chart(fig_bar, width="stretch")
    else:
        st.info("`model_metrics.csv` not found yet. Run Step 4 to generate evaluation metrics.")

    # 3. Actual vs. Predicted Plot
    st.subheader("3. Actual vs. Predicted Forecast")
    if preds_df is not None and "Actual" in preds_df.columns and selected_model in preds_df.columns:
        plot_slice = preds_df.iloc[: min(window_points, len(preds_df))]
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(x=plot_slice.index, y=plot_slice["Actual"], mode="lines", name="Actual"))
        fig_ts.add_trace(
            go.Scatter(
                x=plot_slice.index,
                y=plot_slice[selected_model],
                mode="lines",
                name=selected_model,
                line=dict(dash="dash"),
            )
        )
        fig_ts.update_layout(
            title=f"Actual vs. {selected_model} (First {len(plot_slice)} Test Points)",
            xaxis_title="Timestamp",
            yaxis_title=target_col,
        )
        st.plotly_chart(fig_ts, width="stretch")
    else:
        st.info("Saved test predictions not available for this model yet.")

    # 4. Feature Importance Chart
    st.subheader("4. Feature Importance")
    if imp_df is not None and "feature" in imp_df.columns:
        imp_cols = [c for c in ["mean_abs_shap", "rf_importance", "xgb_importance", "permutation_importance"] if c in imp_df.columns]
        if imp_cols:
            chosen_imp = st.selectbox("Importance Source", options=imp_cols, index=0)
            top_imp = imp_df.sort_values(chosen_imp, ascending=False).head(12).sort_values(chosen_imp, ascending=True)
            fig_imp = px.bar(
                top_imp,
                x=chosen_imp,
                y="feature",
                orientation="h",
                title=f"Top 12 Features ({chosen_imp})",
            )
            st.plotly_chart(fig_imp, width="stretch")
    elif "random_forest" in models and hasattr(models["random_forest"], "feature_importances_"):
        rf_imp_df = (
            pd.DataFrame({
                "feature": feature_cols,
                "rf_importance": models["random_forest"].feature_importances_,
            })
            .sort_values("rf_importance", ascending=False)
            .head(12)
            .sort_values("rf_importance", ascending=True)
        )
        fig_imp = px.bar(
            rf_imp_df,
            x="rf_importance",
            y="feature",
            orientation="h",
            title="Top 12 Features (Random Forest)",
        )
        st.plotly_chart(fig_imp, width="stretch")
    else:
        st.info("Run Step 6 explainability to save `feature_importance.csv`.")

    # 5. Interactive Prediction Form
    st.subheader("5. Make a Custom Load Prediction")
    default_row = clean_df[feature_cols].iloc[-1].copy()
    editable_cols = [
        c for c in [
            "temperature_c",
            "humidity_pct",
            "rainfall_mm",
            "solar_irradiance",
            "tariff_lkr_kwh",
            "Hour of Day",
            "Day of Week",
            f"{target_col}_lag_1",
            f"{target_col}_lag_24",
        ]
        if c in feature_cols
    ]
    if not editable_cols:
        editable_cols = feature_cols[: min(6, len(feature_cols))]

    with st.form("prediction_form"):
        input_vals = {}
        cols_ui = st.columns(3)
        for idx, col in enumerate(editable_cols):
            with cols_ui[idx % 3]:
                base_val = float(default_row[col])
                min_v = float(clean_df[col].min())
                max_v = float(clean_df[col].max())
                if min_v == max_v:
                    min_v, max_v = base_val - 1.0, base_val + 1.0
                step_v = 1.0 if pd.api.types.is_integer_dtype(clean_df[col]) else round((max_v - min_v) / 100.0, 4) or 0.1
                input_vals[col] = st.number_input(
                    label=col,
                    min_value=min_v,
                    max_value=max_v,
                    value=base_val,
                    step=step_v,
                )

        submitted = st.form_submit_button("Predict Electricity Load")

    if submitted:
        row_df = pd.DataFrame([default_row.to_dict()])[feature_cols]
        for k, v in input_vals.items():
            row_df[k] = v
        pred_val = predict_with_model(selected_model, models, row_df)[0]
        st.success(f"**Predicted {target_col} ({selected_model}):** `{pred_val:,.2f}`")


if __name__ == "__main__":
    main()