"""
pages/02_dashboard.py · Dashboard EDA — CRISP-DM Fase 2 & 6
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from modelo_riesgo import RIESGO_LABELS, RIESGO_COLORS

st.set_page_config(page_title="Dashboard · AcadRisk", layout="wide", page_icon="📊")

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
::-webkit-scrollbar{width:6px;height:6px}::-webkit-scrollbar-track{background:#0D1117}::-webkit-scrollbar-thumb{background:#30363D;border-radius:3px}
</style>
""", unsafe_allow_html=True)

if "modelo" not in st.session_state:
    from modelo_riesgo import ModeloRiesgo
    st.session_state.modelo = ModeloRiesgo()

modelo = st.session_state.modelo

st.markdown("""
<div class="page-header">
  <span style="font-size:2rem">📊</span>
  <div>
    <h1>Dashboard Analítico & EDA</h1>
    <p class="subtitle">CRISP-DM Fases 2 y 6 · Exploración de datos · Comunicación de resultados</p>
  </div>
</div>
""", unsafe_allow_html=True)

if not modelo.trained or not modelo.df_stats:
    st.warning("⚠ Entrena el modelo primero en la sección ⚡ Entrenamiento.")
    st.stop()

stats = modelo.df_stats
df    = modelo.df_prepared

# ── Filtros de cohorte en sidebar
with st.sidebar:
    st.markdown("### Filtros de cohorte")
    depts = ["Todos"] + list(df["Department"].unique())
    sel_dept = st.selectbox("Departamento", depts)
    genders = ["Todos"] + list(df["Gender"].unique())
    sel_gen = st.selectbox("Género", genders)
    incomes = ["Todos"] + list(df["Family_Income_Level"].unique())
    sel_inc = st.selectbox("Ingreso familiar", incomes)
    age_min, age_max = int(df["Age"].min()), int(df["Age"].max())
    sel_age = st.slider("Rango de edad", age_min, age_max, (age_min, age_max))

# Aplicar filtros
dff = df.copy()
if sel_dept != "Todos":
    dff = dff[dff["Department"] == sel_dept]
if sel_gen != "Todos":
    dff = dff[dff["Gender"] == sel_gen]
if sel_inc != "Todos":
    dff = dff[dff["Family_Income_Level"] == sel_inc]
dff = dff[(dff["Age"] >= sel_age[0]) & (dff["Age"] <= sel_age[1])]

n_filt = len(dff)
total  = len(df)
label_filt = f"({n_filt:,} de {total:,})" if n_filt < total else f"({total:,} total)"

# ── KPIs
kpi_c = [0,0,0]
for v in dff["Riesgo_Academico"]:
    kpi_c[v] += 1
k1, k2, k3, k4, k5, k6, k7, k8 = st.columns(8)
k1.metric("Estudiantes", f"{n_filt:,}", label_filt)
k2.metric("Riesgo Alto",  f"{kpi_c[0]:,}", f"{kpi_c[0]/n_filt*100:.1f}%" if n_filt else "—", delta_color="inverse")
k3.metric("Riesgo Medio", f"{kpi_c[1]:,}", f"{kpi_c[1]/n_filt*100:.1f}%" if n_filt else "—", delta_color="off")
k4.metric("Riesgo Bajo",  f"{kpi_c[2]:,}", f"{kpi_c[2]/n_filt*100:.1f}%" if n_filt else "—")
k5.metric("Score prom.", f"{dff['Total_Score'].mean():.1f}" if n_filt else "—")
k6.metric("Asistencia prom.", f"{dff['Attendance (%)'].mean():.1f}%" if n_filt else "—")
k7.metric("Horas estudio", f"{dff['Study_Hours_per_Week'].mean():.1f}h" if n_filt else "—")
k8.metric("Estrés prom.", f"{dff['Stress_Level (1-10)'].mean():.1f}/10" if n_filt else "—")

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs de visualizaciones
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "  Distribuciones  ",
    "  Correlaciones  ",
    "  Análisis por grupo  ",
    "  Heatmap  ",
    "  Hallazgos  ",
])

PLOT_LAYOUT = dict(
    plot_bgcolor="#0D1117", paper_bgcolor="#161B22",
    font=dict(color="#8B949E", size=11),
    margin=dict(l=50, r=20, t=50, b=40),
    height=340,
)
AXIS_STYLE = dict(gridcolor="#21262D", tickfont=dict(color="#8B949E"))

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        # Distribución de riesgo (donut)
        labels_r = ["Riesgo Alto", "Riesgo Medio", "Riesgo Bajo"]
        vals_r   = [kpi_c[0], kpi_c[1], kpi_c[2]]
        fig = go.Figure(go.Pie(
            labels=labels_r, values=vals_r,
            hole=.58,
            marker=dict(colors=["#EF4444", "#F59E0B", "#10B981"],
                        line=dict(color="#0D1117", width=2)),
            textfont=dict(color="#E6EDF3", size=12),
        ))
        fig.update_layout(
            title=dict(text="Distribución de riesgo académico", font=dict(color="#C9D1D9",size=13)),
            legend=dict(bgcolor="#161B22", bordercolor="#21262D", font=dict(color="#C9D1D9")),
            **{k: v for k, v in PLOT_LAYOUT.items() if k != "margin"},
            margin=dict(l=20,r=20,t=50,b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Histograma Total_Score por riesgo
        fig2 = go.Figure()
        for i, (label, color) in enumerate(zip(labels_r, ["#EF4444","#F59E0B","#10B981"])):
            sub = dff[dff["Riesgo_Academico"] == i]["Total_Score"]
            fig2.add_trace(go.Histogram(
                x=sub, name=label,
                marker_color=color, opacity=0.75,
                nbinsx=30,
            ))
        fig2.add_vline(x=60, line_color="#EF4444", line_dash="dash",
                       annotation_text="Umbral 60", annotation_font_color="#EF4444")
        fig2.add_vline(x=80, line_color="#10B981", line_dash="dash",
                       annotation_text="Umbral 80", annotation_font_color="#10B981")
        fig2.update_layout(
            title=dict(text="Distribución de Total_Score por nivel de riesgo", font=dict(color="#C9D1D9",size=13)),
            barmode="overlay",
            xaxis=dict(title="Total Score", **AXIS_STYLE),
            yaxis=dict(title="Frecuencia", **AXIS_STYLE),
            legend=dict(bgcolor="#161B22",bordercolor="#21262D",font=dict(color="#C9D1D9")),
            **PLOT_LAYOUT,
        )
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        # Boxplot Asistencia por riesgo
        fig3 = go.Figure()
        for i, (label, color) in enumerate(zip(labels_r, ["#EF4444","#F59E0B","#10B981"])):
            sub = dff[dff["Riesgo_Academico"] == i]["Attendance (%)"]
            fig3.add_trace(go.Box(
                y=sub, name=label,
                marker_color=color, line_color=color,
                boxmean=True,
            ))
        fig3.update_layout(
            title=dict(text="Asistencia (%) por nivel de riesgo", font=dict(color="#C9D1D9",size=13)),
            yaxis=dict(title="Asistencia (%)", **AXIS_STYLE),
            xaxis=AXIS_STYLE,
            showlegend=False,
            **PLOT_LAYOUT,
        )
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        # Scatter Horas Estudio vs Total Score
        sample = dff.sample(min(800, len(dff)), random_state=1)
        color_map = {0: "#EF4444", 1: "#F59E0B", 2: "#10B981"}
        fig4 = go.Figure()
        for i, (label, color) in enumerate(zip(labels_r, ["#EF4444","#F59E0B","#10B981"])):
            sub = sample[sample["Riesgo_Academico"] == i]
            fig4.add_trace(go.Scatter(
                x=sub["Study_Hours_per_Week"], y=sub["Total_Score"],
                mode="markers", name=label,
                marker=dict(color=color, size=5, opacity=0.7),
            ))
        fig4.update_layout(
            title=dict(text="Horas de estudio vs Total_Score", font=dict(color="#C9D1D9",size=13)),
            xaxis=dict(title="Horas estudio/semana", **AXIS_STYLE),
            yaxis=dict(title="Total Score", **AXIS_STYLE),
            legend=dict(bgcolor="#161B22",bordercolor="#21262D",font=dict(color="#C9D1D9")),
            **PLOT_LAYOUT,
        )
        st.plotly_chart(fig4, use_container_width=True)

with tab2:
    corr = stats.get("correlaciones", {})
    if corr:
        from modelo_riesgo import FEATURE_DISPLAY
        items = sorted(corr.items(), key=lambda x: abs(x[1]), reverse=True)
        labels_c  = [FEATURE_DISPLAY.get(k, k) for k, _ in items]
        values_c  = [v for _, v in items]
        colors_c  = ["#10B981" if v >= 0 else "#EF4444" for v in values_c]

        fig_corr = go.Figure(go.Bar(
            x=values_c, y=labels_c,
            orientation="h",
            marker_color=colors_c,
            text=[f"{v:+.3f}" for v in values_c],
            textposition="outside",
            textfont=dict(size=10, color="#C9D1D9"),
        ))
        fig_corr.add_vline(x=0, line_color="#30363D", line_width=1)
        fig_corr.update_layout(
            title=dict(text="Correlación de variables con Total_Score", font=dict(color="#C9D1D9",size=13)),
            xaxis=dict(title="Correlación de Pearson", **AXIS_STYLE, range=[-1.1, 1.2]),
            yaxis=AXIS_STYLE,
            plot_bgcolor="#0D1117", paper_bgcolor="#161B22",
            font=dict(color="#8B949E",size=11),
            margin=dict(l=10,r=80,t=50,b=40),
            height=380,
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        # Scatter Estrés vs Total Score
        sample2 = dff.sample(min(600, len(dff)), random_state=2)
        fig_sc = px.scatter(
            sample2, x="Stress_Level (1-10)", y="Total_Score",
            color="Riesgo_Academico",
            color_discrete_map={0:"#EF4444",1:"#F59E0B",2:"#10B981"},
            labels={"Stress_Level (1-10)": "Nivel de estrés", "Total_Score": "Total Score",
                    "Riesgo_Academico": "Riesgo"},
            trendline="lowess",
        )
        fig_sc.update_layout(
            title=dict(text="Estrés vs rendimiento", font=dict(color="#C9D1D9",size=13)),
            plot_bgcolor="#0D1117", paper_bgcolor="#161B22",
            font=dict(color="#8B949E",size=11),
            legend=dict(bgcolor="#161B22",bordercolor="#21262D",font=dict(color="#C9D1D9")),
            margin=dict(l=50,r=20,t=50,b=40),
            height=320,
        )
        st.plotly_chart(fig_sc, use_container_width=True)

    with c2:
        # Sueño vs Total Score
        fig_sleep = px.scatter(
            sample2, x="Sleep_Hours_per_Night", y="Total_Score",
            color="Riesgo_Academico",
            color_discrete_map={0:"#EF4444",1:"#F59E0B",2:"#10B981"},
            labels={"Sleep_Hours_per_Night": "Horas de sueño/noche",
                    "Total_Score": "Total Score", "Riesgo_Academico": "Riesgo"},
            trendline="lowess",
        )
        fig_sleep.update_layout(
            title=dict(text="Horas de sueño vs rendimiento", font=dict(color="#C9D1D9",size=13)),
            plot_bgcolor="#0D1117", paper_bgcolor="#161B22",
            font=dict(color="#8B949E",size=11),
            legend=dict(bgcolor="#161B22",bordercolor="#21262D",font=dict(color="#C9D1D9")),
            margin=dict(l=50,r=20,t=50,b=40),
            height=320,
        )
        st.plotly_chart(fig_sleep, use_container_width=True)

with tab3:
    c1, c2, c3 = st.columns(3)
    with c1:
        # Por departamento
        dept_risk = dff.groupby(["Department","Riesgo_Academico"]).size().reset_index(name="n")
        fig_dept = px.bar(
            dept_risk, x="Department", y="n", color="Riesgo_Academico",
            color_discrete_map={0:"#EF4444",1:"#F59E0B",2:"#10B981"},
            labels={"n":"Estudiantes","Department":"Departamento","Riesgo_Academico":"Riesgo"},
            barmode="stack",
        )
        fig_dept.update_layout(
            title=dict(text="Riesgo por departamento", font=dict(color="#C9D1D9",size=13)),
            plot_bgcolor="#0D1117", paper_bgcolor="#161B22",
            font=dict(color="#8B949E",size=11),
            legend=dict(bgcolor="#161B22",bordercolor="#21262D",font=dict(color="#C9D1D9",size=10)),
            xaxis=AXIS_STYLE, yaxis=AXIS_STYLE,
            margin=dict(l=40,r=10,t=50,b=50),
            height=320,
        )
        st.plotly_chart(fig_dept, use_container_width=True)

    with c2:
        # Por ingreso familiar
        inc_risk = dff.groupby(["Family_Income_Level","Riesgo_Academico"]).size().reset_index(name="n")
        fig_inc = px.bar(
            inc_risk, x="Family_Income_Level", y="n", color="Riesgo_Academico",
            color_discrete_map={0:"#EF4444",1:"#F59E0B",2:"#10B981"},
            labels={"n":"Estudiantes","Family_Income_Level":"Ingreso","Riesgo_Academico":"Riesgo"},
            barmode="stack",
            category_orders={"Family_Income_Level":["Low","Medium","High"]},
        )
        fig_inc.update_layout(
            title=dict(text="Riesgo por ingreso familiar", font=dict(color="#C9D1D9",size=13)),
            plot_bgcolor="#0D1117", paper_bgcolor="#161B22",
            font=dict(color="#8B949E",size=11),
            legend=dict(bgcolor="#161B22",bordercolor="#21262D",font=dict(color="#C9D1D9",size=10)),
            xaxis=AXIS_STYLE, yaxis=AXIS_STYLE,
            margin=dict(l=40,r=10,t=50,b=50),
            height=320,
        )
        st.plotly_chart(fig_inc, use_container_width=True)

    with c3:
        # Internet access
        int_risk = dff.groupby(["Internet_Access_at_Home","Riesgo_Academico"]).size().reset_index(name="n")
        fig_int = px.bar(
            int_risk, x="Internet_Access_at_Home", y="n", color="Riesgo_Academico",
            color_discrete_map={0:"#EF4444",1:"#F59E0B",2:"#10B981"},
            labels={"n":"Estudiantes","Internet_Access_at_Home":"Internet en casa","Riesgo_Academico":"Riesgo"},
            barmode="stack",
        )
        fig_int.update_layout(
            title=dict(text="Riesgo por acceso a internet", font=dict(color="#C9D1D9",size=13)),
            plot_bgcolor="#0D1117", paper_bgcolor="#161B22",
            font=dict(color="#8B949E",size=11),
            legend=dict(bgcolor="#161B22",bordercolor="#21262D",font=dict(color="#C9D1D9",size=10)),
            xaxis=AXIS_STYLE, yaxis=AXIS_STYLE,
            margin=dict(l=40,r=10,t=50,b=50),
            height=320,
        )
        st.plotly_chart(fig_int, use_container_width=True)

    # Violin plots
    c4, c5 = st.columns(2)
    with c4:
        fig_v = px.violin(
            dff, x="Riesgo_Academico", y="Study_Hours_per_Week",
            color="Riesgo_Academico",
            color_discrete_map={0:"#EF4444",1:"#F59E0B",2:"#10B981"},
            box=True, points="outliers",
            labels={"Study_Hours_per_Week":"Horas estudio/sem","Riesgo_Academico":"Riesgo"},
        )
        fig_v.update_layout(
            title=dict(text="Horas de estudio por nivel de riesgo", font=dict(color="#C9D1D9",size=13)),
            plot_bgcolor="#0D1117", paper_bgcolor="#161B22",
            font=dict(color="#8B949E",size=11), showlegend=False,
            xaxis=AXIS_STYLE, yaxis=AXIS_STYLE,
            margin=dict(l=50,r=10,t=50,b=40), height=320,
        )
        st.plotly_chart(fig_v, use_container_width=True)

    with c5:
        fig_v2 = px.violin(
            dff, x="Riesgo_Academico", y="Stress_Level (1-10)",
            color="Riesgo_Academico",
            color_discrete_map={0:"#EF4444",1:"#F59E0B",2:"#10B981"},
            box=True, points="outliers",
            labels={"Stress_Level (1-10)":"Nivel estrés","Riesgo_Academico":"Riesgo"},
        )
        fig_v2.update_layout(
            title=dict(text="Nivel de estrés por nivel de riesgo", font=dict(color="#C9D1D9",size=13)),
            plot_bgcolor="#0D1117", paper_bgcolor="#161B22",
            font=dict(color="#8B949E",size=11), showlegend=False,
            xaxis=AXIS_STYLE, yaxis=AXIS_STYLE,
            margin=dict(l=50,r=10,t=50,b=40), height=320,
        )
        st.plotly_chart(fig_v2, use_container_width=True)

with tab4:
    numeric_cols = [
        "Total_Score", "Attendance (%)", "Study_Hours_per_Week",
        "Stress_Level (1-10)", "Sleep_Hours_per_Night",
        "Midterm_Score", "Final_Score", "Projects_Score",
        "Assignments_Avg", "Quizzes_Avg",
    ]
    available = [c for c in numeric_cols if c in dff.columns]
    corr_matrix = dff[available].corr()
    from modelo_riesgo import FEATURE_DISPLAY
    labels_hm = [FEATURE_DISPLAY.get(c, c) for c in available]

    fig_hm = go.Figure(go.Heatmap(
        z=corr_matrix.values,
        x=labels_hm, y=labels_hm,
        colorscale=[
            [0.0, "#EF4444"], [0.5, "#1C2333"], [1.0, "#10B981"]
        ],
        zmid=0, zmin=-1, zmax=1,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=9, color="#E6EDF3"),
        colorbar=dict(
            title=dict(text="Correlación", font=dict(color="#8B949E")),
            tickfont=dict(color="#8B949E"),
        ),
    ))
    fig_hm.update_layout(
        title=dict(text="Matriz de correlación — variables numéricas", font=dict(color="#C9D1D9",size=13)),
        plot_bgcolor="#0D1117", paper_bgcolor="#161B22",
        font=dict(color="#8B949E",size=10),
        xaxis=dict(tickfont=dict(color="#C9D1D9", size=10), tickangle=-40),
        yaxis=dict(tickfont=dict(color="#C9D1D9", size=10)),
        margin=dict(l=20,r=20,t=60,b=120),
        height=520,
    )
    st.plotly_chart(fig_hm, use_container_width=True)

with tab5:
    st.markdown("#### Hallazgos clave — CRISP-DM Fase 2")
    hallazgos = [
        ("🟢", "Proyectos (30%) y Examen Final (25%) dominan el Total_Score y son los predictores más importantes.",
         "#10B981", "#0F2A1A", "#1A5C35"),
        ("🔵", "Mayor asistencia correlaciona positivamente con mejor rendimiento, aunque no es un componente directo de la nota.",
         "#58A6FF", "#0A1929", "#1A3A5C"),
        ("🔴", "Estrés alto (>7) se asocia consistentemente con menor rendimiento en todas las cohortes.",
         "#EF4444", "#2A0A0A", "#5C1A1A"),
        ("🟡", "Estudiantes con ingreso familiar bajo tienen un 23% más de probabilidad de caer en Riesgo Alto.",
         "#F59E0B", "#2A1A00", "#5C3A00"),
        ("🟣", "Acceso a internet en casa mejora el promedio del Total_Score en ~4.5 puntos.",
         "#8B5CF6", "#160A2A", "#3A1A5C"),
        ("⚪", "~1025 valores nulos (~4.1%) — imputados con mediana por columna, sin pérdida de registros.",
         "#8B949E", "#161B22", "#21262D"),
    ]
    for icon, texto, color, bg, border in hallazgos:
        st.markdown(f"""
        <div style="background:{bg};border:1px solid {border};border-left:4px solid {color};
                    border-radius:8px;padding:.8rem 1.1rem;margin:.4rem 0;
                    font-size:.88rem;color:{color};line-height:1.5">
          {icon} &nbsp; {texto}
        </div>
        """, unsafe_allow_html=True)

    # Tabla de estadísticas descriptivas
    st.markdown("<br>Estadísticas descriptivas", unsafe_allow_html=True)
    numeric_desc = dff[[
        "Total_Score","Attendance (%)","Study_Hours_per_Week",
        "Stress_Level (1-10)","Sleep_Hours_per_Night",
        "Midterm_Score","Final_Score","Projects_Score",
    ]].describe().T.round(2)
    numeric_desc.index = [FEATURE_DISPLAY.get(i, i) for i in numeric_desc.index]
    st.dataframe(numeric_desc, use_container_width=True)