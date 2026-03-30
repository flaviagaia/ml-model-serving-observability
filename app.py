from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.model_training import ARTIFACTS_DIR, train_and_persist_model


st.set_page_config(page_title="ML Model Serving Observability", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background: #09111f; color: #ecf2ff; }
    [data-testid="stSidebar"] { background: #0d1728; }
    </style>
    """,
    unsafe_allow_html=True,
)

artifacts = train_and_persist_model()
metrics = pd.read_csv(artifacts.metrics_path)
sample_payload = pd.read_csv(artifacts.sample_path)

st.title("ML Model Serving Observability")
st.caption(
    "Projeto de observabilidade para serving de modelo com FastAPI, Prometheus e Grafana."
)

with st.expander("Arquitetura técnica", expanded=False):
    st.markdown(
        """
        ```mermaid
        flowchart LR
            A["Client / Load Generator"] --> B["FastAPI model server"]
            B --> C["/predict endpoint"]
            B --> D["/metrics endpoint"]
            D --> E["Prometheus scrape"]
            E --> F["Grafana dashboards"]
        ```
        """
    )
    st.markdown(
        """
        **Componentes**
        - `FastAPI`: inference API e health endpoint.
        - `prometheus_client`: instrumentação e exportação de métricas.
        - `Prometheus`: coleta e armazenamento temporal.
        - `Grafana`: visualização, exploração e alertas.
        """
    )

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{metrics.iloc[0]['accuracy'] * 100:.2f}%")
col2.metric("Macro F1", f"{metrics.iloc[0]['macro_f1'] * 100:.2f}%")
col3.metric("Classes", str(int(metrics.iloc[0]["classes"])))
col4.metric("Features", str(int(metrics.iloc[0]["features"])))

tab1, tab2, tab3 = st.tabs(["Modelo", "Payload demo", "Observabilidade"])

with tab1:
    st.subheader("Métricas de treino")
    st.dataframe(metrics, use_container_width=True)

with tab2:
    st.subheader("Amostra de payload para o endpoint /predict")
    st.dataframe(sample_payload.drop(columns=["target"]), use_container_width=True)
    st.code(
        sample_payload.drop(columns=["target"]).head(1).to_json(orient="records", indent=2),
        language="json",
    )

with tab3:
    st.subheader("Métricas expostas ao Prometheus")
    observability_metrics = pd.DataFrame(
        [
            {"metric": "model_inference_requests_total", "purpose": "request throughput and error rate"},
            {"metric": "model_inference_latency_seconds", "purpose": "latency histogram and SLI tracking"},
            {"metric": "model_predictions_total", "purpose": "class distribution monitoring"},
            {"metric": "model_prediction_confidence", "purpose": "confidence drift and quality inspection"},
            {"metric": "model_metadata", "purpose": "served model metadata"},
        ]
    )
    st.dataframe(observability_metrics, use_container_width=True)
    fig = px.bar(
        observability_metrics,
        x="metric",
        y=[1] * len(observability_metrics),
        title="Camadas de instrumentação da API",
    )
    fig.update_layout(
        paper_bgcolor="#09111f",
        plot_bgcolor="#09111f",
        font={"color": "#ecf2ff"},
        yaxis_title="enabled",
        xaxis_title="metric",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
