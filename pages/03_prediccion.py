"""
pages/03_prediccion.py · Predicción Individual + SHAP + Simulador de intervención
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from modelo_riesgo import RIESGO_LABELS, RIESGO_COLORS, ENC_MAPS

st.set_page_config(page_title="Predicción · AcadRisk", layout="wide", page_icon="🎯")
st.markdown("""
<style>
html,[class*="css"]{font-family:'Inter',sans-serif}
[data-testid="stAppViewContainer"]>.main{background:#0D1117}
.block-container{padding:1.5rem 2rem 2rem!important;max-width:1400px}
.risk-card{background:#161B22;border:1px solid #21262D;border-radius:10px;padding:1rem 1.25rem}
.page-header{display:flex;align-items:center;gap:12px;padding-bottom:1rem;border-bottom:1px solid #21262D;margin-bottom:1.5rem}
.page-header h1{font-size:1.35rem;font-weight:600;color:#E6EDF3;margin:0}
.page-header .subtitle{font-size:.8rem;color:#8B949E;margin:0}
.stButton>button{border-radius:8px!important}
::-webkit-scrollbar{width:6px}::-webkit-scrollbar-track{background:#0D1117}::-webkit-scrollbar-thumb{background:#30363D;border-radius:3px}
</style>
""", unsafe_allow_html=True)

if "modelo" not in st.session_state:
    from modelo_riesgo import ModeloRiesgo
    st.session_state.modelo = ModeloRiesgo()

modelo = st.session_state.modelo

st.markdown("""
<div class="page-header">
  <span style="font-size:2rem">🎯</span>
  <div>
    <h1>Predicción Individual en Tiempo Real</h1>
    <p class="subtitle">Ingresa los datos de un estudiante para predecir su nivel de riesgo académico · SHAP explainability</p>
  </div>
</div>
""", unsafe_allow_html=True)

if not modelo.trained:
    st.warning("⚠ Entrena el modelo primero en la sección ⚡ Entrenamiento.")
    st.stop()

# ── Selector de modelo
modelo_nombre = st.selectbox(
    "Modelo predictivo activo",
    options=list(modelo.modelos.keys()),
    index=list(modelo.modelos.keys()).index("Random Forest") if "Random Forest" in modelo.modelos else 0,
)

# ── Perfiles rápidos
col_pre1, col_pre2, col_pre3 = st.columns([1,1,6])
with col_pre1:
    if st.button("Perfil típico"):
        st.session_state["perfil"] = "tipico"
with col_pre2:
    if st.button("Perfil en riesgo"):
        st.session_state["perfil"] = "riesgo"

perfil = st.session_state.get("perfil", None)
PERFILES = {
    "tipico": {
        "Attendance (%)": 82.0, "Midterm_Score": 70.0, "Final_Score": 68.0,
        "Assignments_Avg": 72.0, "Quizzes_Avg": 65.0, "Participation_Score": 70.0,
        "Projects_Score": 68.0, "Study_Hours_per_Week": 15.0,
        "Stress_Level (1-10)": 5, "Sleep_Hours_per_Night": 7.0, "Age": 20,
        "Gender": "Male", "Department": "Engineering",
        "Extracurricular_Activities": "Yes", "Internet_Access_at_Home": "Yes",
        "Parent_Education_Level": "Bachelor", "Family_Income_Level": "Medium",
    },
    "riesgo": {
        "Attendance (%)": 55.0, "Midterm_Score": 42.0, "Final_Score": 40.0,
        "Assignments_Avg": 38.0, "Quizzes_Avg": 36.0, "Participation_Score": 28.0,
        "Projects_Score": 33.0, "Study_Hours_per_Week": 4.0,
        "Stress_Level (1-10)": 9, "Sleep_Hours_per_Night": 4.5, "Age": 22,
        "Gender": "Male", "Department": "Business",
        "Extracurricular_Activities": "No", "Internet_Access_at_Home": "No",
        "Parent_Education_Level": "High School", "Family_Income_Level": "Low",
    },
}
defaults = PERFILES.get(perfil, PERFILES["tipico"])

st.markdown("---")
st.markdown("#### Variables del estudiante")

# ── Formulario en 3 columnas
col_a, col_b, col_c = st.columns([1, 1, 1])

with col_a:
    st.markdown("**📚 Académicas**")
    attendance = st.slider("Asistencia (%)", 0.0, 100.0, float(defaults["Attendance (%)"]), step=0.5)
    midterm    = st.slider("Nota Parcial", 0.0, 100.0, float(defaults["Midterm_Score"]), step=0.5)
    final_sc   = st.slider("Nota Final", 0.0, 100.0, float(defaults["Final_Score"]), step=0.5)
    assignments= st.slider("Promedio Tareas", 0.0, 100.0, float(defaults["Assignments_Avg"]), step=0.5)
    quizzes    = st.slider("Promedio Quizzes", 0.0, 100.0, float(defaults["Quizzes_Avg"]), step=0.5)
    participac = st.slider("Participación", 0.0, 100.0, float(defaults["Participation_Score"]), step=0.5)
    projects   = st.slider("Nota Proyectos", 0.0, 100.0, float(defaults["Projects_Score"]), step=0.5)

with col_b:
    st.markdown("**🧠 Conductuales**")
    study_h   = st.slider("Horas Estudio/Sem", 0.0, 40.0, float(defaults["Study_Hours_per_Week"]), step=0.5)
    stress    = st.slider("Nivel Estrés (1-10)", 1, 10, int(defaults["Stress_Level (1-10)"]))
    sleep_h   = st.slider("Horas Sueño/Noche", 3.0, 12.0, float(defaults["Sleep_Hours_per_Night"]), step=0.25)
    age       = st.slider("Edad", 16, 35, int(defaults["Age"]))
    st.markdown("**🏠 Socioeconómicas**")
    gender    = st.selectbox("Género", ["Male","Female"],
                              index=["Male","Female"].index(defaults["Gender"]))
    department= st.selectbox("Departamento", ["CS","Engineering","Mathematics","Business"],
                              index=["CS","Engineering","Mathematics","Business"].index(defaults["Department"]))
    extracurr = st.selectbox("Actividades Extra.", ["Yes","No"],
                              index=["Yes","No"].index(defaults["Extracurricular_Activities"]))
    internet  = st.selectbox("Internet en Casa", ["Yes","No"],
                              index=["Yes","No"].index(defaults["Internet_Access_at_Home"]))
    parent_edu= st.selectbox("Educación Padres", ["High School","Bachelor","Master","PhD"],
                              index=["High School","Bachelor","Master","PhD"].index(defaults["Parent_Education_Level"]))
    family_inc= st.selectbox("Ingreso Familiar", ["Low","Medium","High"],
                              index=["Low","Medium","High"].index(defaults["Family_Income_Level"]))

# ── Construir vector de features
enc = {
    "Gender_enc":    ENC_MAPS["Gender"][gender],
    "Dept_enc":      ENC_MAPS["Department"][department],
    "Extracurr_enc": ENC_MAPS["Extracurricular_Activities"][extracurr],
    "Internet_enc":  ENC_MAPS["Internet_Access_at_Home"][internet],
    "ParentEdu_enc": ENC_MAPS["Parent_Education_Level"][parent_edu],
    "FamilyInc_enc": ENC_MAPS["Family_Income_Level"][family_inc],
}
vals = {
    "Attendance (%)": attendance, "Midterm_Score": midterm,
    "Final_Score": final_sc, "Assignments_Avg": assignments,
    "Quizzes_Avg": quizzes, "Participation_Score": participac,
    "Projects_Score": projects, "Study_Hours_per_Week": study_h,
    "Stress_Level (1-10)": stress, "Sleep_Hours_per_Night": sleep_h,
    "Age": age, **enc,
}

# ── Panel de resultado (columna derecha)
with col_c:
    pred, probs = modelo.predecir(vals, modelo_nombre)
    riesgo_label = RIESGO_LABELS[pred]
    riesgo_color = RIESGO_COLORS[pred]
    confianza    = max(probs) * 100

    border_map = {0: "#EF4444", 1: "#F59E0B", 2: "#10B981"}
    bg_map     = {0: "rgba(239,68,68,.10)", 1: "rgba(245,158,11,.10)", 2: "rgba(16,185,129,.10)"}
    icon_map   = {0: "🔴", 1: "🟡", 2: "🟢"}

    st.markdown(f"""
    <div style="background:{bg_map[pred]};border:2px solid {riesgo_color};
                border-radius:14px;padding:1.5rem;text-align:center;margin-top:1.5rem">
      <div style="font-size:2.5rem">{icon_map[pred]}</div>
      <div style="font-size:.75rem;color:{riesgo_color};font-weight:700;
                  text-transform:uppercase;letter-spacing:.1em;margin:.4rem 0">
        Nivel de riesgo predicho
      </div>
      <div style="font-size:1.8rem;font-weight:700;color:{riesgo_color}">
        {riesgo_label}
      </div>
      <div style="font-size:.85rem;color:#8B949E;margin:.3rem 0">
        Confianza: <b style="color:{riesgo_color}">{confianza:.1f}%</b>
      </div>
      <div style="font-size:.78rem;color:#6E7681;margin-top:.2rem">
        Modelo: {modelo_nombre}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Probabilidades
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Distribución de probabilidades**")
    fig_prob = go.Figure()
    for i, (label, color) in enumerate(zip(
        ["Riesgo Alto","Riesgo Medio","Riesgo Bajo"],
        ["#EF4444","#F59E0B","#10B981"]
    )):
        fig_prob.add_trace(go.Bar(
            x=[probs[i]*100], y=[label],
            orientation="h",
            marker_color=color,
            marker_opacity=1.0 if i == pred else 0.4,
            text=f"{probs[i]*100:.1f}%",
            textposition="outside",
            textfont=dict(color="#C9D1D9", size=12),
        ))
    fig_prob.update_layout(
        plot_bgcolor="#0D1117", paper_bgcolor="#161B22",
        font=dict(color="#8B949E"),
        xaxis=dict(range=[0,115], showticklabels=False, showgrid=False),
        yaxis=dict(tickfont=dict(color="#C9D1D9", size=12)),
        showlegend=False, barmode="overlay",
        margin=dict(l=10,r=60,t=10,b=10),
        height=160,
    )
    st.plotly_chart(fig_prob, use_container_width=True)

    # Total Score estimado
    total_est = (0.15 * midterm + 0.25 * final_sc + 0.15 * assignments +
                 0.10 * quizzes + 0.05 * participac + 0.30 * projects)
    st.markdown(f"""
    <div style="background:#161B22;border:1px solid #21262D;border-radius:8px;
                padding:.75rem 1rem;text-align:center;margin-top:.5rem">
      <div style="font-size:.7rem;color:#8B949E;text-transform:uppercase;letter-spacing:.06em">
        Total Score estimado
      </div>
      <div style="font-size:1.5rem;font-weight:600;color:#E6EDF3">{total_est:.1f}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── SHAP Explainability
st.markdown("#### SHAP — Explicabilidad del modelo")
st.markdown("<span style='color:#8B949E;font-size:.85rem'>¿Por qué el modelo tomó esta decisión?</span>",
            unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def get_shap_explainer(_modelo_obj, modelo_nombre):
    try:
        import shap
        m = _modelo_obj.modelos[modelo_nombre]
        if modelo_nombre in ["Random Forest", "Decision Tree"]:
            explainer = shap.TreeExplainer(m)
        elif modelo_nombre == "XGBoost":
            explainer = shap.TreeExplainer(m)
        else:
            bg_data = np.zeros((1, len(_modelo_obj.feature_names)))
            explainer = shap.KernelExplainer(m.predict_proba, bg_data)
        return explainer, True
    except Exception as e:
        return None, False

with st.spinner("Calculando SHAP values..."):
    X_instance = np.array([float(vals.get(f, 0.0)) for f in modelo.feature_names]).reshape(1, -1)
    if modelo_nombre == "Logistic Regression":
        X_instance_scaled = modelo.scaler.transform(X_instance)
    else:
        X_instance_scaled = X_instance

    explainer, shap_ok = get_shap_explainer(modelo, modelo_nombre)

if shap_ok and explainer is not None:
    try:
        import shap
        if modelo_nombre == "Logistic Regression":
            shap_vals = explainer.shap_values(X_instance_scaled)
        else:
            shap_vals = explainer.shap_values(X_instance)

        # shap_vals puede ser lista (multi-clase) o array
        if isinstance(shap_vals, list):
            sv = shap_vals[pred][0]
        else:
            sv = shap_vals[0] if shap_vals.ndim == 2 else shap_vals[0, :, pred]

        from modelo_riesgo import FEATURE_DISPLAY
        feat_labels = [FEATURE_DISPLAY.get(f, f) for f in modelo.feature_names]

        # Ordenar por magnitud absoluta
        sorted_idx = np.argsort(np.abs(sv))[::-1][:12]
        sv_top  = sv[sorted_idx]
        fl_top  = [feat_labels[i] for i in sorted_idx]
        fv_top  = [X_instance[0][i] for i in sorted_idx]

        colors_shap = ["#EF4444" if v > 0 else "#10B981" for v in sv_top]

        fig_shap = go.Figure(go.Bar(
            x=sv_top[::-1], y=[f"{fl_top[i]} = {fv_top[i]:.1f}" for i in range(len(fl_top))][::-1],
            orientation="h",
            marker_color=colors_shap[::-1],
            text=[f"{v:+.4f}" for v in sv_top[::-1]],
            textposition="outside",
            textfont=dict(size=10, color="#C9D1D9"),
        ))
        fig_shap.add_vline(x=0, line_color="#30363D", line_width=1)
        fig_shap.update_layout(
            title=dict(
                text=f"SHAP values — impacto en clase '{riesgo_label}' (rojo=aumenta riesgo, verde=reduce)",
                font=dict(color="#C9D1D9", size=13),
            ),
            plot_bgcolor="#0D1117", paper_bgcolor="#161B22",
            font=dict(color="#8B949E", size=11),
            xaxis=dict(title="SHAP value", gridcolor="#21262D", tickfont=dict(color="#8B949E")),
            yaxis=dict(tickfont=dict(color="#C9D1D9", size=11)),
            margin=dict(l=10, r=90, t=60, b=40),
            height=420,
        )
        st.plotly_chart(fig_shap, use_container_width=True)

        st.markdown(f"""
        <div style="background:rgba(88,166,255,.07);border:1px solid rgba(88,166,255,.2);
                    border-radius:8px;padding:.75rem 1rem;font-size:.83rem;color:#79C0FF">
          ℹ️ &nbsp;<b>Cómo leer este gráfico:</b> Barras rojas = variables que <b>aumentan</b> 
          la probabilidad de este nivel de riesgo. Barras verdes = variables que la <b>reducen</b>. 
          La longitud representa la magnitud del impacto.
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.info(f"SHAP no disponible para este modelo en esta configuración: {e}")
else:
    # Fallback: feature importance del RF como proxy
    st.markdown("""
    <div style="background:rgba(245,158,11,.08);border:1px solid rgba(245,158,11,.25);
                border-radius:8px;padding:.75rem 1rem;font-size:.83rem;color:#FCD34D">
      ⚠️ &nbsp;SHAP no disponible para este modelo. Mostrando importancia global del Random Forest como referencia.
    </div>
    """, unsafe_allow_html=True)
    from modelo_riesgo import FEATURE_DISPLAY
    imp = modelo.importancia
    top_imp = list(imp.items())[:10]
    feat_l = [FEATURE_DISPLAY.get(k, k) for k, _ in top_imp]
    feat_v = [v for _, v in top_imp]
    fig_fi = go.Figure(go.Bar(
        x=feat_v[::-1], y=feat_l[::-1], orientation="h",
        marker_color="#58A6FF",
        text=[f"{v:.4f}" for v in feat_v[::-1]],
        textposition="outside", textfont=dict(size=10, color="#C9D1D9"),
    ))
    fig_fi.update_layout(
        plot_bgcolor="#0D1117", paper_bgcolor="#161B22",
        font=dict(color="#8B949E", size=11),
        xaxis=dict(gridcolor="#21262D", tickfont=dict(color="#8B949E")),
        yaxis=dict(tickfont=dict(color="#C9D1D9", size=11)),
        margin=dict(l=10, r=80, t=20, b=30),
        height=320,
    )
    st.plotly_chart(fig_fi, use_container_width=True)

# ── Simulador de intervención
st.markdown("---")
st.markdown("#### Simulador de Intervención Académica")
st.markdown("<span style='color:#8B949E;font-size:.85rem'>¿Cómo cambia el riesgo si el estudiante mejora ciertas variables?</span>",
            unsafe_allow_html=True)

col_s1, col_s2 = st.columns(2)
with col_s1:
    new_study   = st.slider("Nuevas horas de estudio/sem",   0.0, 40.0, min(study_h + 5, 40.0), step=0.5)
    new_stress  = st.slider("Nuevo nivel de estrés",          1, 10, max(1, stress - 2))
    new_attend  = st.slider("Nueva asistencia (%)",           0.0, 100.0, min(attendance + 10, 100.0), step=0.5)
    new_projects= st.slider("Nueva nota de proyectos",        0.0, 100.0, min(projects + 10, 100.0), step=0.5)

vals_new = vals.copy()
vals_new["Study_Hours_per_Week"]  = new_study
vals_new["Stress_Level (1-10)"]   = new_stress
vals_new["Attendance (%)"]        = new_attend
vals_new["Projects_Score"]        = new_projects

pred_new, probs_new = modelo.predecir(vals_new, modelo_nombre)
riesgo_new   = RIESGO_LABELS[pred_new]
color_new    = RIESGO_COLORS[pred_new]
conf_new     = max(probs_new) * 100

with col_s2:
    changed = pred_new != pred
    change_icon = "📈 Mejora detectada" if pred_new > pred else ("📉 Empeora" if pred_new < pred else "↔ Sin cambio")
    change_color = "#10B981" if pred_new > pred else ("#EF4444" if pred_new < pred else "#8B949E")

    st.markdown(f"""
    <div style="background:{bg_map[pred_new]};border:2px solid {color_new};
                border-radius:14px;padding:1.2rem;text-align:center;margin-top:1rem">
      <div style="font-size:.7rem;color:{color_new};font-weight:700;
                  text-transform:uppercase;letter-spacing:.1em">
        Riesgo proyectado
      </div>
      <div style="font-size:1.5rem;font-weight:700;color:{color_new};margin:.3rem 0">
        {riesgo_new}
      </div>
      <div style="font-size:.85rem;color:#8B949E">
        Confianza: <b style="color:{color_new}">{conf_new:.1f}%</b>
      </div>
      <div style="font-size:.9rem;color:{change_color};font-weight:600;margin-top:.5rem">
        {change_icon}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Comparación probabilidades
    fig_comp = go.Figure()
    labels_r = ["Riesgo Alto","Riesgo Medio","Riesgo Bajo"]
    colors_r = ["#EF4444","#F59E0B","#10B981"]
    x_pos = np.arange(3)
    for i, (lbl, col) in enumerate(zip(labels_r, colors_r)):
        fig_comp.add_trace(go.Bar(
            name="Antes", x=[lbl], y=[probs[i]*100],
            marker_color=col, opacity=0.45,
            legendgroup="antes", showlegend=(i==0),
        ))
        fig_comp.add_trace(go.Bar(
            name="Después", x=[lbl], y=[probs_new[i]*100],
            marker_color=col, opacity=1.0,
            legendgroup="despues", showlegend=(i==0),
        ))
    fig_comp.update_layout(
        barmode="group",
        title=dict(text="Antes vs después de intervención (%)", font=dict(color="#C9D1D9",size=12)),
        plot_bgcolor="#0D1117", paper_bgcolor="#161B22",
        font=dict(color="#8B949E",size=10),
        yaxis=dict(title="%", gridcolor="#21262D", range=[0,115]),
        xaxis=dict(tickfont=dict(color="#C9D1D9")),
        legend=dict(bgcolor="#161B22",bordercolor="#21262D",font=dict(color="#C9D1D9",size=10)),
        margin=dict(l=40,r=10,t=40,b=30),
        height=240,
    )
    st.plotly_chart(fig_comp, use_container_width=True)
