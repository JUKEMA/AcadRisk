"""
pages/04_alertas.py · Tabla de Alertas — estudiantes en riesgo rankeados
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from modelo_riesgo import RIESGO_LABELS, RIESGO_COLORS

st.set_page_config(page_title="Alertas · AcadRisk", layout="wide", page_icon="🚨")
st.markdown("""
<style>
html,[class*="css"]{font-family:'Inter',sans-serif}
[data-testid="stAppViewContainer"]>.main{background:#0D1117}
.block-container{padding:1.5rem 2rem 2rem!important;max-width:1400px}
.risk-card{background:#161B22;border:1px solid #21262D;border-radius:10px;padding:1rem 1.25rem}
.page-header{display:flex;align-items:center;gap:12px;padding-bottom:1rem;border-bottom:1px solid #21262D;margin-bottom:1.5rem}
.page-header h1{font-size:1.35rem;font-weight:600;color:#E6EDF3;margin:0}
.page-header .subtitle{font-size:.8rem;color:#8B949E;margin:0}
::-webkit-scrollbar{width:6px}::-webkit-scrollbar-track{background:#0D1117}::-webkit-scrollbar-thumb{background:#30363D;border-radius:3px}
</style>
""", unsafe_allow_html=True)

if "modelo" not in st.session_state:
    from modelo_riesgo import ModeloRiesgo
    st.session_state.modelo = ModeloRiesgo()

modelo = st.session_state.modelo

st.markdown("""
<div class="page-header">
  <span style="font-size:2rem">🚨</span>
  <div>
    <h1>Tabla de Alertas — Estudiantes en Riesgo</h1>
    <p class="subtitle">Predicción sobre todo el dataset · Filtros · Exportación · Acción institucional</p>
  </div>
</div>
""", unsafe_allow_html=True)

if not modelo.trained:
    st.warning("⚠ Entrena el modelo primero en la sección ⚡ Entrenamiento.")
    st.stop()

# ── Selector modelo
modelo_nombre = st.selectbox(
    "Modelo para generar alertas",
    options=list(modelo.modelos.keys()),
    index=list(modelo.modelos.keys()).index("Random Forest") if "Random Forest" in modelo.modelos else 0,
)

with st.spinner("Generando predicciones sobre el dataset completo..."):
    df_pred = modelo.predecir_dataset(modelo_nombre)

if df_pred is None:
    st.error("No se pudo generar la tabla de predicciones.")
    st.stop()

# ── Filtros
with st.expander("🔍 Filtros", expanded=True):
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        sel_riesgo = st.multiselect(
            "Nivel de riesgo",
            options=["Riesgo Alto","Riesgo Medio","Riesgo Bajo"],
            default=["Riesgo Alto","Riesgo Medio"],
        )
    with f2:
        depts = ["Todos"] + sorted(df_pred["Department"].unique().tolist())
        sel_dept = st.selectbox("Departamento", depts)
    with f3:
        min_conf = st.slider("Confianza mínima (%)", 0, 100, 60, step=5)
    with f4:
        top_n = st.selectbox("Mostrar top", [50, 100, 200, 500, "Todos"], index=1)

# Aplicar filtros
dff = df_pred.copy()
if sel_riesgo:
    dff = dff[dff["Riesgo_Label"].isin(sel_riesgo)]
if sel_dept != "Todos":
    dff = dff[dff["Department"] == sel_dept]
dff = dff[dff["Confianza (%)"] >= min_conf]
dff = dff.sort_values(["Riesgo_Pred", "P_Riesgo_Alto"], ascending=[True, False])
if top_n != "Todos":
    dff = dff.head(int(top_n))

# ── KPIs de la vista filtrada
k1, k2, k3, k4 = st.columns(4)
total_filt = len(dff)
n_alto  = (dff["Riesgo_Pred"] == 0).sum()
n_medio = (dff["Riesgo_Pred"] == 1).sum()
n_bajo  = (dff["Riesgo_Pred"] == 2).sum()
k1.metric("Estudiantes en vista", f"{total_filt:,}")
k2.metric("🔴 Riesgo Alto",  f"{n_alto:,}",  f"{n_alto/total_filt*100:.1f}%" if total_filt else "—", delta_color="inverse")
k3.metric("🟡 Riesgo Medio", f"{n_medio:,}", f"{n_medio/total_filt*100:.1f}%" if total_filt else "—", delta_color="off")
k4.metric("🟢 Riesgo Bajo",  f"{n_bajo:,}",  f"{n_bajo/total_filt*100:.1f}%" if total_filt else "—")

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabla principal
display_cols = [
    "Student_ID",
    "First_Name" if "First_Name" in dff.columns else None,
    "Last_Name"  if "Last_Name"  in dff.columns else None,
    "Department", "Age", "Gender",
    "Attendance (%)", "Total_Score",
    "Study_Hours_per_Week", "Stress_Level (1-10)",
    "Riesgo_Label", "P_Riesgo_Alto", "Confianza (%)",
]
display_cols = [c for c in display_cols if c and c in dff.columns]
df_show = dff[display_cols].copy()

# Estilizar la tabla
def color_riesgo(val):
    if val == "Riesgo Alto":
        return "color: #EF4444; font-weight: 600"
    elif val == "Riesgo Medio":
        return "color: #F59E0B; font-weight: 600"
    elif val == "Riesgo Bajo":
        return "color: #10B981; font-weight: 600"
    return ""

st.dataframe(
    df_show.style.applymap(color_riesgo, subset=["Riesgo_Label"])
               .format({"Attendance (%)": "{:.1f}", "Total_Score": "{:.1f}",
                        "P_Riesgo_Alto": "{:.1f}%", "Confianza (%)": "{:.1f}%",
                        "Study_Hours_per_Week": "{:.1f}"}),
    use_container_width=True,
    height=420,
    hide_index=True,
)

# ── Exportar CSV
csv_data = df_show.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Exportar tabla como CSV",
    data=csv_data,
    file_name=f"alertas_academicas_{modelo_nombre.replace(' ','_').lower()}.csv",
    mime="text/csv",
)

st.markdown("---")

# ── Gráficos adicionales
c1, c2 = st.columns(2)

with c1:
    # Top 20 en riesgo más alto con confianza
    top20 = dff[dff["Riesgo_Pred"] == 0].head(20).copy()
    if len(top20) > 0:
        if "First_Name" in top20.columns and "Last_Name" in top20.columns:
            top20["Nombre"] = top20["First_Name"] + " " + top20["Last_Name"]
        elif "Student_ID" in top20.columns:
            top20["Nombre"] = "ID-" + top20["Student_ID"].astype(str)
        else:
            top20["Nombre"] = [f"Est. {i+1}" for i in range(len(top20))]

        fig = go.Figure(go.Bar(
            x=top20["P_Riesgo_Alto"].values[::-1],
            y=top20["Nombre"].values[::-1],
            orientation="h",
            marker_color="#EF4444",
            text=[f"{v:.1f}%" for v in top20["P_Riesgo_Alto"].values[::-1]],
            textposition="outside",
            textfont=dict(size=9, color="#C9D1D9"),
        ))
        fig.update_layout(
            title=dict(text="Top 20 estudiantes con mayor P(Riesgo Alto)", font=dict(color="#C9D1D9",size=12)),
            plot_bgcolor="#0D1117", paper_bgcolor="#161B22",
            font=dict(color="#8B949E",size=10),
            xaxis=dict(range=[0,115], gridcolor="#21262D", tickfont=dict(color="#8B949E"), title="%"),
            yaxis=dict(tickfont=dict(color="#C9D1D9",size=10)),
            margin=dict(l=10,r=70,t=50,b=30),
            height=460,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No hay estudiantes en Riesgo Alto con los filtros actuales.")

with c2:
    # Distribución por departamento en la vista filtrada
    dept_risk = dff.groupby(["Department","Riesgo_Label"]).size().reset_index(name="n")
    if len(dept_risk) > 0:
        import plotly.express as px
        fig2 = px.bar(
            dept_risk, x="Department", y="n", color="Riesgo_Label",
            color_discrete_map={
                "Riesgo Alto":"#EF4444",
                "Riesgo Medio":"#F59E0B",
                "Riesgo Bajo":"#10B981",
            },
            labels={"n":"Estudiantes","Department":"Departamento","Riesgo_Label":"Nivel"},
            barmode="stack",
        )
        fig2.update_layout(
            title=dict(text="Distribución de riesgo por departamento (vista filtrada)",
                       font=dict(color="#C9D1D9",size=12)),
            plot_bgcolor="#0D1117", paper_bgcolor="#161B22",
            font=dict(color="#8B949E",size=10),
            legend=dict(bgcolor="#161B22",bordercolor="#21262D",font=dict(color="#C9D1D9",size=10)),
            xaxis=dict(gridcolor="#21262D",tickfont=dict(color="#8B949E")),
            yaxis=dict(gridcolor="#21262D",tickfont=dict(color="#8B949E")),
            margin=dict(l=40,r=10,t=50,b=50),
            height=320,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Scatter Total Score vs Confianza
    sample_a = dff.sample(min(500, len(dff)), random_state=7)
    color_map_r = {"Riesgo Alto":"#EF4444","Riesgo Medio":"#F59E0B","Riesgo Bajo":"#10B981"}
    fig3 = go.Figure()
    for label, color in color_map_r.items():
        sub = sample_a[sample_a["Riesgo_Label"] == label]
        if len(sub):
            fig3.add_trace(go.Scatter(
                x=sub["Total_Score"], y=sub["Confianza (%)"],
                mode="markers", name=label,
                marker=dict(color=color, size=5, opacity=0.7),
            ))
    fig3.update_layout(
        title=dict(text="Total Score vs confianza del modelo", font=dict(color="#C9D1D9",size=12)),
        plot_bgcolor="#0D1117", paper_bgcolor="#161B22",
        font=dict(color="#8B949E",size=10),
        xaxis=dict(title="Total Score", gridcolor="#21262D",tickfont=dict(color="#8B949E")),
        yaxis=dict(title="Confianza (%)", gridcolor="#21262D",tickfont=dict(color="#8B949E")),
        legend=dict(bgcolor="#161B22",bordercolor="#21262D",font=dict(color="#C9D1D9",size=10)),
        margin=dict(l=50,r=10,t=50,b=40),
        height=300,
    )
    st.plotly_chart(fig3, use_container_width=True)
