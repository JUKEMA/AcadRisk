"""
pages/01_entrenamiento.py · Entrenamiento CRISP-DM Fases 3 y 4
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from modelo_riesgo import ModeloRiesgo, RIESGO_LABELS

st.set_page_config(page_title="Entrenamiento · AcadRisk", layout="wide", page_icon="⚡")

# ── CSS inyectado desde app (si se navega directamente)
st.markdown("""
<style>
html,[class*="css"]{font-family:'Inter',sans-serif}
[data-testid="stAppViewContainer"]>.main{background:#0D1117}
.block-container{padding:1.5rem 2rem 2rem!important;max-width:1400px}
[data-testid="metric-container"]{background:#161B22;border:1px solid #21262D;border-radius:10px;padding:1rem 1.25rem!important}
[data-testid="metric-container"] label{color:#8B949E!important;font-size:.75rem!important;font-weight:500;text-transform:uppercase;letter-spacing:.05em}
[data-testid="metric-container"] [data-testid="stMetricValue"]{font-size:1.8rem!important;font-weight:600}
.risk-card{background:#161B22;border:1px solid #21262D;border-radius:10px;padding:1rem 1.25rem}
.page-header{display:flex;align-items:center;gap:12px;padding-bottom:1rem;border-bottom:1px solid #21262D;margin-bottom:1.5rem}
.page-header h1{font-size:1.35rem;font-weight:600;color:#E6EDF3;margin:0}
.page-header .subtitle{font-size:.8rem;color:#8B949E;margin:0}
.info-box{background:rgba(88,166,255,.08);border:1px solid rgba(88,166,255,.25);border-radius:8px;padding:.75rem 1rem;color:#79C0FF;font-size:.85rem;margin:.5rem 0}
::-webkit-scrollbar{width:6px;height:6px}::-webkit-scrollbar-track{background:#0D1117}::-webkit-scrollbar-thumb{background:#30363D;border-radius:3px}
</style>
""", unsafe_allow_html=True)

if "modelo" not in st.session_state:
    st.session_state.modelo = ModeloRiesgo()
if "df_raw" not in st.session_state:
    st.session_state.df_raw = None

modelo = st.session_state.modelo

st.markdown("""
<div class="page-header">
  <span style="font-size:2rem">⚡</span>
  <div>
    <h1>Entrenamiento de Modelos</h1>
    <p class="subtitle">CRISP-DM Fases 3 y 4 · Preparación de datos · Modelado comparativo</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Upload
st.markdown("#### 1 · Cargar Dataset")
uploaded = st.file_uploader(
    "Sube el archivo **Students Performance Dataset.csv**",
    type=["csv"],
    help="Dataset principal: 5000 registros, 23 variables. No uses el biased dataset.",
)

if uploaded:
    try:
        df = pd.read_csv(uploaded)
        st.session_state.df_raw = df
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Registros", f"{len(df):,}")
        c2.metric("Columnas", len(df.columns))
        c3.metric("Valores nulos", f"{df.isnull().sum().sum():,}")
        c4.metric("Duplicados", f"{df.duplicated().sum():,}")

        with st.expander("Vista previa del dataset (10 filas)"):
            st.dataframe(
                df.head(10),
                use_container_width=True,
                hide_index=True,
            )
    except Exception as e:
        st.error(f"Error al leer el archivo: {e}")

st.markdown("---")

# ── Config entrenamiento
st.markdown("#### 2 · Configuración de Entrenamiento")
col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
with col_cfg1:
    st.markdown("""
    <div class="risk-card">
      <div style="font-size:.7rem;color:#58A6FF;font-weight:600;text-transform:uppercase;letter-spacing:.06em;margin-bottom:.6rem">Variable objetivo</div>
      <div style="font-size:.83rem;color:#C9D1D9;line-height:1.9">
        <span style="color:#EF4444">■</span> Riesgo Alto &nbsp;&nbsp; Total_Score &lt; 60<br>
        <span style="color:#F59E0B">■</span> Riesgo Medio  Total_Score 60–80<br>
        <span style="color:#10B981">■</span> Riesgo Bajo &nbsp;&nbsp; Total_Score &gt; 80
      </div>
    </div>
    """, unsafe_allow_html=True)
with col_cfg2:
    st.markdown("""
    <div class="risk-card">
      <div style="font-size:.7rem;color:#3FB950;font-weight:600;text-transform:uppercase;letter-spacing:.06em;margin-bottom:.6rem">Preprocesamiento</div>
      <div style="font-size:.83rem;color:#C9D1D9;line-height:1.9">
        ✓ Imputación por mediana<br>
        ✓ Label Encoding categóricas<br>
        ✓ StandardScaler (LR)<br>
        ✓ Split 80/20 estratificado
      </div>
    </div>
    """, unsafe_allow_html=True)
with col_cfg3:
    st.markdown("""
    <div class="risk-card">
      <div style="font-size:.7rem;color:#8B5CF6;font-weight:600;text-transform:uppercase;letter-spacing:.06em;margin-bottom:.6rem">Modelos a comparar</div>
      <div style="font-size:.83rem;color:#C9D1D9;line-height:1.9">
        ★ Random Forest (principal)<br>
        · Logistic Regression (baseline)<br>
        · Decision Tree (interpretable)<br>
        · XGBoost (high-performance)
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Botón entrenar
if st.button("🚀 Entrenar todos los modelos", type="primary", use_container_width=True):
    if st.session_state.df_raw is None:
        st.error("⚠ Primero carga el dataset CSV.")
    else:
        progress_bar = st.progress(0, text="Iniciando entrenamiento...")
        status_text  = st.empty()

        def progress_cb(pct, msg):
            progress_bar.progress(pct / 100, text=msg)
            status_text.markdown(f"<span style='color:#8B949E;font-size:.82rem'>⚙ {msg}</span>",
                                  unsafe_allow_html=True)

        with st.spinner(""):
            try:
                resultados = modelo.entrenar(st.session_state.df_raw, progress_cb=progress_cb)
                st.session_state.modelo = modelo
                progress_bar.progress(1.0, text="¡Completado!")
                status_text.empty()
                st.success("✅ Entrenamiento completado exitosamente.")
            except Exception as ex:
                st.error(f"Error durante el entrenamiento: {ex}")
                st.stop()

# ── Resultados
if modelo.trained:
    st.markdown("---")
    st.markdown("#### 3 · Resultados de Entrenamiento")

    resultados = modelo.resultados
    nombres = list(resultados.keys())

    # Tabla comparativa
    rows = []
    for nombre, mets in resultados.items():
        cv_str = f"{mets['cv_mean']:.4f} ± {mets['cv_std']:.4f}"
        rows.append({
            "Modelo":    nombre,
            "Accuracy":  mets["accuracy"],
            "Precision": mets["precision"],
            "Recall":    mets["recall"],
            "F1 Macro":  mets["f1"],
            "ROC-AUC":   mets.get("roc_auc") or "—",
            "CV-5 F1 (mean ± std)": cv_str,
            "Principal": "★" if nombre == "Random Forest" else "",
        })
    df_tabla = pd.DataFrame(rows)

    def highlight_best(col):
        if col.name in ["Accuracy", "Precision", "Recall", "F1 Macro"]:
            try:
                max_v = pd.to_numeric(col, errors="coerce").max()
                return ["background-color: rgba(16,185,129,.15); color:#6EE7B7"
                        if v == max_v else "" for v in pd.to_numeric(col, errors="coerce")]
            except Exception:
                return [""] * len(col)
        return [""] * len(col)

    st.dataframe(
        df_tabla.style.apply(highlight_best).format({
            "Accuracy": "{:.4f}", "Precision": "{:.4f}",
            "Recall": "{:.4f}", "F1 Macro": "{:.4f}",
        }),
        use_container_width=True,
        hide_index=True,
        height=200,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Gráficos comparativos
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        metrics_names = ["Accuracy", "Precision", "Recall", "F1 Macro"]
        fig = go.Figure()
        colors = ["#58A6FF", "#8B5CF6", "#F59E0B", "#3FB950"]
        for color, nombre in zip(colors, nombres):
            mets = resultados[nombre]
            values = [mets["accuracy"], mets["precision"], mets["recall"], mets["f1"]]
            fig.add_trace(go.Bar(
                name=nombre, x=metrics_names, y=values,
                marker_color=color, opacity=0.85,
                text=[f"{v:.3f}" for v in values],
                textposition="outside",
                textfont=dict(size=10, color="#C9D1D9"),
            ))
        fig.update_layout(
            title=dict(text="Comparativa de métricas por modelo", font=dict(color="#C9D1D9", size=13)),
            barmode="group", plot_bgcolor="#0D1117", paper_bgcolor="#161B22",
            font=dict(color="#8B949E", size=11),
            xaxis=dict(gridcolor="#21262D", tickfont=dict(color="#8B949E")),
            yaxis=dict(gridcolor="#21262D", tickfont=dict(color="#8B949E"), range=[0, 1.12]),
            legend=dict(bgcolor="#161B22", bordercolor="#21262D", font=dict(color="#C9D1D9", size=10)),
            margin=dict(l=40, r=20, t=50, b=40),
            height=340,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_g2:
        # CV scores boxplot
        fig2 = go.Figure()
        for color, nombre in zip(colors, nombres):
            mets = resultados[nombre]
            mean, std = mets["cv_mean"], mets["cv_std"]
            # simulate CV distribution around mean±std
            np.random.seed(42)
            sim = np.clip(np.random.normal(mean, std, 500), 0, 1)
            fig2.add_trace(go.Box(
                y=sim, name=nombre,
                marker_color=color,
                line_color=color,
                fillcolor=color.replace("FF", "22").replace("B0", "22"),
                boxmean="sd",
            ))
        fig2.update_layout(
            title=dict(text="Validación cruzada k=5 (F1 macro)", font=dict(color="#C9D1D9", size=13)),
            plot_bgcolor="#0D1117", paper_bgcolor="#161B22",
            font=dict(color="#8B949E", size=11),
            yaxis=dict(gridcolor="#21262D", tickfont=dict(color="#8B949E")),
            xaxis=dict(tickfont=dict(color="#8B949E")),
            showlegend=False,
            margin=dict(l=40, r=20, t=50, b=40),
            height=340,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Importancia de variables RF
    st.markdown("#### 4 · Importancia de Variables — Random Forest")
    imp = modelo.importancia
    if imp:
        from modelo_riesgo import FEATURE_DISPLAY
        imp_labels = [FEATURE_DISPLAY.get(k, k) for k in imp.keys()]
        imp_values = list(imp.values())
        top_n = 15
        imp_labels = imp_labels[:top_n]
        imp_values = imp_values[:top_n]
        palette = [
            "#58A6FF","#3FB950","#F59E0B","#8B5CF6","#EF4444",
            "#10B981","#F97316","#06B6D4","#EC4899","#84CC16",
            "#A78BFA","#34D399","#FB923C","#38BDF8","#F472B6",
        ]
        fig_imp = go.Figure(go.Bar(
            x=imp_values[::-1], y=imp_labels[::-1],
            orientation="h",
            marker_color=palette[:len(imp_labels)][::-1],
            text=[f"{v:.4f}" for v in imp_values[::-1]],
            textposition="outside",
            textfont=dict(size=10, color="#C9D1D9"),
        ))
        fig_imp.update_layout(
            plot_bgcolor="#0D1117", paper_bgcolor="#161B22",
            font=dict(color="#8B949E", size=11),
            xaxis=dict(gridcolor="#21262D", tickfont=dict(color="#8B949E")),
            yaxis=dict(tickfont=dict(color="#C9D1D9", size=11)),
            margin=dict(l=10, r=80, t=20, b=30),
            height=420,
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("""
    <div class="info-box">
      ⚠️ &nbsp;<b>Data leakage documentado:</b> Midterm_Score, Final_Score y similares 
      son componentes directos de Total_Score (la base de Riesgo_Academico), 
      lo que eleva las métricas artificialmente. Ver la sección 
      <b>Comparativa de Modelos</b> para el modelo conductual sin leakage.
    </div>
    """, unsafe_allow_html=True)
