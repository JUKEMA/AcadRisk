"""
pages/05_modelos.py · Comparativa de Modelos — con análisis de Data Leakage
CRISP-DM Fase 5: Evaluación
Compara los 4 modelos completos vs. un modelo conductual (sin leakage académico).
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from modelo_riesgo import RIESGO_LABELS, RIESGO_COLORS, BEHAVIORAL_COLS, FEATURE_DISPLAY

st.set_page_config(page_title="Comparativa Modelos · AcadRisk", layout="wide", page_icon="🔬")

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
.warn-box{background:rgba(245,158,11,.08);border:1px solid rgba(245,158,11,.25);border-radius:8px;padding:.75rem 1rem;color:#FCD34D;font-size:.85rem;margin:.5rem 0}
::-webkit-scrollbar{width:6px;height:6px}::-webkit-scrollbar-track{background:#0D1117}::-webkit-scrollbar-thumb{background:#30363D;border-radius:3px}
</style>
""", unsafe_allow_html=True)

if "modelo" not in st.session_state:
    from modelo_riesgo import ModeloRiesgo
    st.session_state.modelo = ModeloRiesgo()

modelo = st.session_state.modelo

st.markdown("""
<div class="page-header">
  <span style="font-size:2rem">🔬</span>
  <div>
    <h1>Comparativa de Modelos & Análisis de Data Leakage</h1>
    <p class="subtitle">CRISP-DM Fase 5 · Evaluación · Modelo completo vs. modelo conductual · Matrices de confusión · ROC-AUC</p>
  </div>
</div>
""", unsafe_allow_html=True)

if not modelo.trained:
    st.warning("⚠ Entrena el modelo primero en la sección ⚡ Entrenamiento.")
    st.stop()

# ── Aviso metodológico
st.markdown("""
<div class="warn-box">
  ⚠️ &nbsp;<b>Nota de Data Leakage documentada:</b> El modelo completo usa variables como
  <code>Midterm_Score</code>, <code>Final_Score</code> y <code>Assignments_Avg</code> que son
  componentes directos de <code>Total_Score</code> (base de la variable objetivo).
  Esto eleva artificialmente las métricas. El <b>Modelo Conductual</b> (sección inferior) usa
  solo variables socioeconómicas y de comportamiento para una comparación metodológicamente limpia.
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

resultados = modelo.resultados
nombres    = list(resultados.keys())
COLORS_MOD = {"Random Forest": "#58A6FF", "Logistic Regression": "#8B5CF6",
               "Decision Tree": "#F59E0B", "XGBoost": "#3FB950"}

# ─────────────────────────────────────────────────────────────────
# SECCIÓN 1: TABLA COMPARATIVA COMPLETA
# ─────────────────────────────────────────────────────────────────
st.markdown("#### 1 · Tabla comparativa — Modelo completo (todas las variables)")

rows = []
for nombre in nombres:
    m = resultados[nombre]
    rows.append({
        "Modelo":       nombre,
        "Accuracy":     m["accuracy"],
        "Precision":    m["precision"],
        "Recall":       m["recall"],
        "F1 Macro":     m["f1"],
        "ROC-AUC":      m.get("roc_auc") or 0,
        "CV-5 F1 mean": m["cv_mean"],
        "CV-5 F1 std":  m["cv_std"],
        "Principal":    "★" if nombre == "Random Forest" else "",
    })
df_tabla = pd.DataFrame(rows)

def highlight_best(col):
    if col.name in ["Accuracy", "Precision", "Recall", "F1 Macro", "ROC-AUC", "CV-5 F1 mean"]:
        try:
            max_v = pd.to_numeric(col, errors="coerce").max()
            return ["background-color: rgba(16,185,129,.15); color:#6EE7B7"
                    if v == max_v else "" for v in pd.to_numeric(col, errors="coerce")]
        except Exception:
            return [""] * len(col)
    return [""] * len(col)

st.dataframe(
    df_tabla.style.apply(highlight_best).format({
        "Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}",
        "F1 Macro": "{:.4f}", "ROC-AUC": "{:.4f}",
        "CV-5 F1 mean": "{:.4f}", "CV-5 F1 std": "{:.4f}",
    }),
    use_container_width=True,
    hide_index=True,
    height=210,
)

# ─────────────────────────────────────────────────────────────────
# SECCIÓN 2: RADAR + BARRAS
# ─────────────────────────────────────────────────────────────────
st.markdown("#### 2 · Comparativa visual de métricas")
col_r1, col_r2 = st.columns(2)

with col_r1:
    # Radar chart
    cats = ["Accuracy", "Precision", "Recall", "F1 Macro", "ROC-AUC"]
    fig_radar = go.Figure()
    for nombre in nombres:
        m = resultados[nombre]
        vals_r = [m["accuracy"], m["precision"], m["recall"], m["f1"], m.get("roc_auc") or 0]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals_r + [vals_r[0]],
            theta=cats + [cats[0]],
            name=nombre,
            line_color=COLORS_MOD.get(nombre, "#888"),
            fill="toself",
            fillcolor=COLORS_MOD.get(nombre, "#888").replace("FF", "18").replace("B0", "18"),
            opacity=0.85,
        ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor="#161B22",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#21262D",
                            tickfont=dict(color="#8B949E", size=9), tickvals=[0.25,0.5,0.75,1.0]),
            angularaxis=dict(tickfont=dict(color="#C9D1D9", size=10), gridcolor="#21262D"),
        ),
        paper_bgcolor="#161B22",
        legend=dict(bgcolor="#161B22", bordercolor="#21262D", font=dict(color="#C9D1D9", size=10)),
        margin=dict(l=40, r=40, t=40, b=40),
        height=320,
        title=dict(text="Radar de métricas por modelo", font=dict(color="#C9D1D9", size=13)),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

with col_r2:
    # Grouped bar: F1 vs Accuracy
    fig_bar = go.Figure()
    for nombre in nombres:
        m = resultados[nombre]
        fig_bar.add_trace(go.Bar(
            name=nombre,
            x=["Accuracy", "F1 Macro", "ROC-AUC", "CV-5 F1"],
            y=[m["accuracy"], m["f1"], m.get("roc_auc") or 0, m["cv_mean"]],
            marker_color=COLORS_MOD.get(nombre, "#888"),
            opacity=0.85,
            text=[f"{v:.3f}" for v in [m["accuracy"], m["f1"], m.get("roc_auc") or 0, m["cv_mean"]]],
            textposition="outside",
            textfont=dict(size=9, color="#C9D1D9"),
        ))
    fig_bar.update_layout(
        barmode="group",
        title=dict(text="Métricas clave por modelo", font=dict(color="#C9D1D9", size=13)),
        plot_bgcolor="#0D1117", paper_bgcolor="#161B22",
        font=dict(color="#8B949E", size=11),
        yaxis=dict(gridcolor="#21262D", range=[0, 1.12], tickfont=dict(color="#8B949E")),
        xaxis=dict(tickfont=dict(color="#C9D1D9")),
        legend=dict(bgcolor="#161B22", bordercolor="#21262D", font=dict(color="#C9D1D9", size=10)),
        margin=dict(l=40, r=20, t=50, b=30),
        height=320,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ─────────────────────────────────────────────────────────────────
# SECCIÓN 3: MATRICES DE CONFUSIÓN
# ─────────────────────────────────────────────────────────────────
st.markdown("#### 3 · Matrices de confusión")
st.markdown("<span style='color:#8B949E;font-size:.85rem'>Filas = clase real · Columnas = clase predicha · Diagonal = predicciones correctas</span>",
            unsafe_allow_html=True)

class_labels = ["Riesgo Alto", "Riesgo Medio", "Riesgo Bajo"]
cols_cm = st.columns(len(nombres))

for col_cm, nombre in zip(cols_cm, nombres):
    with col_cm:
        cm = np.array(resultados[nombre]["cm"])
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
        text_vals = [[f"{cm[i][j]}<br><span style='font-size:.7rem'>({cm_norm[i][j]*100:.0f}%)</span>"
                      for j in range(3)] for i in range(3)]
        fig_cm = go.Figure(go.Heatmap(
            z=cm_norm,
            x=["Alto\n(pred)", "Medio\n(pred)", "Bajo\n(pred)"],
            y=["Alto\n(real)", "Medio\n(real)", "Bajo\n(real)"],
            text=text_vals,
            texttemplate="%{text}",
            colorscale=[[0, "#0D1117"], [0.5, "#1F3A5F"], [1, "#58A6FF"]],
            showscale=False,
            xgap=3, ygap=3,
        ))
        fig_cm.update_layout(
            title=dict(text=nombre, font=dict(color="#C9D1D9", size=12)),
            plot_bgcolor="#0D1117", paper_bgcolor="#161B22",
            font=dict(color="#8B949E", size=10),
            xaxis=dict(tickfont=dict(color="#8B949E", size=9), side="bottom"),
            yaxis=dict(tickfont=dict(color="#8B949E", size=9), autorange="reversed"),
            margin=dict(l=60, r=10, t=50, b=50),
            height=260,
        )
        st.plotly_chart(fig_cm, use_container_width=True)

# ─────────────────────────────────────────────────────────────────
# SECCIÓN 4: ROC CURVES (simuladas desde proba almacenadas)
# ─────────────────────────────────────────────────────────────────
st.markdown("#### 4 · Curvas ROC (One-vs-Rest, macro)")

try:
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    fig_roc = go.Figure()
    fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                      line=dict(color="#30363D", width=1, dash="dash"))

    for nombre in nombres:
        m     = resultados[nombre]
        y_t   = np.array(m["y_test"])
        y_p   = np.array(m["y_proba"])
        y_bin = label_binarize(y_t, classes=[0, 1, 2])

        # macro average
        fpr_all, tpr_all = [], []
        for i in range(3):
            fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], y_p[:, i])
            fpr_all.append(fpr_i)
            tpr_all.append(tpr_i)

        mean_fpr = np.linspace(0, 1, 200)
        mean_tpr = np.zeros_like(mean_fpr)
        for fpr_i, tpr_i in zip(fpr_all, tpr_all):
            mean_tpr += np.interp(mean_fpr, fpr_i, tpr_i)
        mean_tpr /= 3
        roc_auc_v = auc(mean_fpr, mean_tpr)

        fig_roc.add_trace(go.Scatter(
            x=mean_fpr, y=mean_tpr,
            mode="lines",
            name=f"{nombre} (AUC={roc_auc_v:.3f})",
            line=dict(color=COLORS_MOD.get(nombre, "#888"), width=2),
        ))

    fig_roc.update_layout(
        title=dict(text="Curvas ROC — comparativa de modelos (macro OvR)", font=dict(color="#C9D1D9", size=13)),
        plot_bgcolor="#0D1117", paper_bgcolor="#161B22",
        font=dict(color="#8B949E", size=11),
        xaxis=dict(title="False Positive Rate", gridcolor="#21262D", range=[0, 1],
                   tickfont=dict(color="#8B949E")),
        yaxis=dict(title="True Positive Rate", gridcolor="#21262D", range=[0, 1.02],
                   tickfont=dict(color="#8B949E")),
        legend=dict(bgcolor="#161B22", bordercolor="#21262D", font=dict(color="#C9D1D9", size=11)),
        margin=dict(l=60, r=20, t=60, b=50),
        height=380,
    )
    st.plotly_chart(fig_roc, use_container_width=True)

except Exception as e:
    st.info(f"No se pudieron graficar las curvas ROC: {e}")

# ─────────────────────────────────────────────────────────────────
# SECCIÓN 5: MODELO CONDUCTUAL (sin leakage)
# ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### 5 · Modelo Conductual — sin Data Leakage")

st.markdown("""
<div class="info-box">
  🧪 &nbsp;<b>¿Qué es el modelo conductual?</b> Se entrena usando <b>solo variables conductuales y
  socioeconómicas</b> (asistencia, horas de estudio, estrés, sueño, ingreso familiar, etc.),
  excluyendo scores académicos directos (Midterm, Final, Assignments, Quizzes, Projects).
  Esto refleja lo que un sistema de alerta <b>temprana real</b> dispondría <i>antes</i> de los exámenes.
</div>
""", unsafe_allow_html=True)

# Mostrar columnas usadas
behavioral_display = [FEATURE_DISPLAY.get(c, c) for c in BEHAVIORAL_COLS]
st.markdown(
    "<div style='margin:.5rem 0 1rem'><span style='color:#8B949E;font-size:.82rem'>"
    "Variables utilizadas: </span>"
    + " · ".join(f"<code>{c}</code>" for c in behavioral_display)
    + "</div>",
    unsafe_allow_html=True,
)

if st.button("⚡ Entrenar modelo conductual (RF)", type="primary"):
    with st.spinner("Entrenando modelo conductual..."):
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
            from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                         recall_score, roc_auc_score, confusion_matrix)

            df = modelo.df_prepared.copy()
            available = [c for c in BEHAVIORAL_COLS if c in df.columns]
            X_b = df[available].values
            y_b = df["Riesgo_Academico"].values

            X_tr, X_te, y_tr, y_te = train_test_split(
                X_b, y_b, test_size=0.2, random_state=42, stratify=y_b)

            rf_b = RandomForestClassifier(
                n_estimators=200, max_depth=12, random_state=42,
                n_jobs=-1, class_weight="balanced")
            rf_b.fit(X_tr, y_tr)
            y_pred_b  = rf_b.predict(X_te)
            y_proba_b = rf_b.predict_proba(X_te)

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_sc = cross_val_score(rf_b, X_b, y_b, cv=cv, scoring="f1_macro")

            roc_auc_b = round(float(roc_auc_score(
                y_te, y_proba_b, multi_class="ovr", average="macro")), 4)

            st.session_state["modelo_conductual"] = {
                "accuracy":  round(accuracy_score(y_te, y_pred_b), 4),
                "precision": round(precision_score(y_te, y_pred_b, average="macro", zero_division=0), 4),
                "recall":    round(recall_score(y_te, y_pred_b, average="macro", zero_division=0), 4),
                "f1":        round(f1_score(y_te, y_pred_b, average="macro", zero_division=0), 4),
                "roc_auc":   roc_auc_b,
                "cv_mean":   round(float(cv_sc.mean()), 4),
                "cv_std":    round(float(cv_sc.std()), 4),
                "cm":        confusion_matrix(y_te, y_pred_b, labels=[0,1,2]).tolist(),
                "importancia": dict(zip(available, rf_b.feature_importances_.tolist())),
                "n_features": len(available),
            }
            st.success("✅ Modelo conductual entrenado.")
        except Exception as ex:
            st.error(f"Error: {ex}")

if "modelo_conductual" in st.session_state:
    mc = st.session_state["modelo_conductual"]
    rf_full = resultados.get("Random Forest", {})

    st.markdown("##### Comparativa: RF Completo vs. RF Conductual")

    # Delta table
    delta_acc = mc["accuracy"] - rf_full["accuracy"]
    delta_f1  = mc["f1"]       - rf_full["f1"]
    delta_roc = mc["roc_auc"]  - rf_full.get("roc_auc", 0)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Accuracy — Completo",    f"{rf_full['accuracy']:.4f}")
    c2.metric("Accuracy — Conductual",  f"{mc['accuracy']:.4f}",
              delta=f"{delta_acc:+.4f}", delta_color="normal")
    c3.metric("F1 Macro — Completo",    f"{rf_full['f1']:.4f}")
    c4.metric("F1 Macro — Conductual",  f"{mc['f1']:.4f}",
              delta=f"{delta_f1:+.4f}", delta_color="normal")
    c5.metric("ROC-AUC — Completo",     f"{rf_full.get('roc_auc', 0):.4f}")
    c6.metric("ROC-AUC — Conductual",   f"{mc['roc_auc']:.4f}",
              delta=f"{delta_roc:+.4f}", delta_color="normal")

    st.markdown("<br>", unsafe_allow_html=True)

    col_c1, col_c2 = st.columns(2)

    with col_c1:
        # Comparativa barras lado a lado
        metricas = ["Accuracy", "F1 Macro", "ROC-AUC", "CV-5 F1"]
        vals_full = [rf_full["accuracy"], rf_full["f1"],
                     rf_full.get("roc_auc", 0), rf_full["cv_mean"]]
        vals_cond = [mc["accuracy"], mc["f1"], mc["roc_auc"], mc["cv_mean"]]

        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Bar(
            name="RF Completo (con leakage)",
            x=metricas, y=vals_full,
            marker_color="#58A6FF", opacity=0.9,
            text=[f"{v:.3f}" for v in vals_full],
            textposition="outside", textfont=dict(size=10, color="#C9D1D9"),
        ))
        fig_cmp.add_trace(go.Bar(
            name="RF Conductual (sin leakage)",
            x=metricas, y=vals_cond,
            marker_color="#F59E0B", opacity=0.9,
            text=[f"{v:.3f}" for v in vals_cond],
            textposition="outside", textfont=dict(size=10, color="#C9D1D9"),
        ))
        fig_cmp.update_layout(
            barmode="group",
            title=dict(text="RF Completo vs. RF Conductual", font=dict(color="#C9D1D9", size=13)),
            plot_bgcolor="#0D1117", paper_bgcolor="#161B22",
            font=dict(color="#8B949E", size=11),
            yaxis=dict(gridcolor="#21262D", range=[0, 1.15], tickfont=dict(color="#8B949E")),
            xaxis=dict(tickfont=dict(color="#C9D1D9")),
            legend=dict(bgcolor="#161B22", bordercolor="#21262D", font=dict(color="#C9D1D9", size=10)),
            margin=dict(l=40, r=20, t=50, b=30),
            height=320,
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

    with col_c2:
        # Importancia de variables en el modelo conductual
        imp_c = mc["importancia"]
        imp_sorted = dict(sorted(imp_c.items(), key=lambda x: x[1], reverse=True))
        labels_c = [FEATURE_DISPLAY.get(k, k) for k in imp_sorted.keys()]
        vals_c   = list(imp_sorted.values())

        palette = ["#F59E0B","#10B981","#58A6FF","#8B5CF6","#EF4444",
                   "#F97316","#06B6D4","#EC4899","#84CC16","#A78BFA","#34D399","#FB923C"]

        fig_imp_c = go.Figure(go.Bar(
            x=vals_c[::-1], y=labels_c[::-1],
            orientation="h",
            marker_color=palette[:len(labels_c)][::-1],
            text=[f"{v:.4f}" for v in vals_c[::-1]],
            textposition="outside",
            textfont=dict(size=10, color="#C9D1D9"),
        ))
        fig_imp_c.update_layout(
            title=dict(text="Importancia — modelo conductual", font=dict(color="#C9D1D9", size=13)),
            plot_bgcolor="#0D1117", paper_bgcolor="#161B22",
            font=dict(color="#8B949E", size=11),
            xaxis=dict(gridcolor="#21262D", tickfont=dict(color="#8B949E")),
            yaxis=dict(tickfont=dict(color="#C9D1D9", size=10)),
            margin=dict(l=10, r=80, t=50, b=30),
            height=320,
        )
        st.plotly_chart(fig_imp_c, use_container_width=True)

    # Matriz de confusión del modelo conductual
    st.markdown("##### Matriz de confusión — modelo conductual")
    cm_c      = np.array(mc["cm"])
    cm_c_norm = cm_c.astype(float) / (cm_c.sum(axis=1, keepdims=True) + 1e-9)
    text_c    = [[f"{cm_c[i][j]}<br><span style='font-size:.7rem'>({cm_c_norm[i][j]*100:.0f}%)</span>"
                  for j in range(3)] for i in range(3)]

    fig_cm_c = go.Figure(go.Heatmap(
        z=cm_c_norm,
        x=["Alto (pred)", "Medio (pred)", "Bajo (pred)"],
        y=["Alto (real)", "Medio (real)", "Bajo (real)"],
        text=text_c,
        texttemplate="%{text}",
        colorscale=[[0, "#0D1117"], [0.5, "#3A2B00"], [1, "#F59E0B"]],
        showscale=False,
        xgap=3, ygap=3,
    ))
    fig_cm_c.update_layout(
        title=dict(text="Modelo conductual — solo variables conductuales/socioeconómicas",
                   font=dict(color="#C9D1D9", size=12)),
        plot_bgcolor="#0D1117", paper_bgcolor="#161B22",
        font=dict(color="#8B949E", size=10),
        xaxis=dict(tickfont=dict(color="#8B949E")),
        yaxis=dict(tickfont=dict(color="#8B949E"), autorange="reversed"),
        margin=dict(l=100, r=20, t=60, b=60),
        height=300,
    )
    st.plotly_chart(fig_cm_c, use_container_width=True)

    st.markdown(f"""
    <div class="info-box">
      📊 &nbsp;<b>Interpretación:</b> La caída de métricas entre el modelo completo y el conductual
      revela la magnitud real del data leakage. Con {mc['n_features']} variables conductuales,
      el modelo logra un F1 macro de <b>{mc['f1']:.4f}</b> vs <b>{rf_full['f1']:.4f}</b>
      del modelo completo. Esta diferencia de <b>{abs(delta_f1):.4f}</b> representa el
      "boost artificial" producido por incluir componentes del score objetivo como predictores.
    </div>
    """, unsafe_allow_html=True)
