"""
pages/06_metodologia.py · Metodología CRISP-DM aplicada al proyecto
Explica cada fase con detalle institucional, decisiones técnicas y justificaciones.
"""

import streamlit as st
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

st.set_page_config(page_title="Metodología · AcadRisk", layout="wide", page_icon="📖")

st.markdown("""
<style>
html,[class*="css"]{font-family:'Inter',sans-serif}
[data-testid="stAppViewContainer"]>.main{background:#0D1117}
.block-container{padding:1.5rem 2rem 2rem!important;max-width:1400px}
.risk-card{background:#161B22;border:1px solid #21262D;border-radius:10px;padding:1rem 1.25rem}
.page-header{display:flex;align-items:center;gap:12px;padding-bottom:1rem;border-bottom:1px solid #21262D;margin-bottom:1.5rem}
.page-header h1{font-size:1.35rem;font-weight:600;color:#E6EDF3;margin:0}
.page-header .subtitle{font-size:.8rem;color:#8B949E;margin:0}
.info-box{background:rgba(88,166,255,.08);border:1px solid rgba(88,166,255,.25);border-radius:8px;padding:.75rem 1rem;color:#79C0FF;font-size:.85rem;margin:.5rem 0}
.fase-card{border-radius:12px;padding:1.25rem 1.5rem;margin-bottom:1rem;border:1px solid}
::-webkit-scrollbar{width:6px;height:6px}::-webkit-scrollbar-track{background:#0D1117}::-webkit-scrollbar-thumb{background:#30363D;border-radius:3px}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="page-header">
  <span style="font-size:2rem">📖</span>
  <div>
    <h1>Metodología CRISP-DM aplicada al proyecto</h1>
    <p class="subtitle">Cross Industry Standard Process for Data Mining · 6 fases · MD 801 · UDeC 2026</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# Resumen visual del ciclo CRISP-DM
# ─────────────────────────────────────────────────────────────────
st.markdown("#### Ciclo CRISP-DM — visión general")

fases_info = [
    {
        "num": "1", "titulo": "Comprensión del negocio",
        "color": "#58A6FF", "bg": "rgba(88,166,255,.07)", "border": "rgba(88,166,255,.25)",
        "objetivo": "Definir el problema institucional y los objetivos del proyecto.",
        "actividades": [
            "Identificar el objetivo principal: detectar estudiantes en riesgo académico antes del corte final.",
            "Definir criterio de éxito: modelo con F1 macro ≥ 0.75 sobre datos reales.",
            "Establecer la variable objetivo: Riesgo_Academico (3 clases) derivada de Total_Score.",
            "Determinar el alcance: 5,000 estudiantes, 23 variables, datos de un semestre.",
        ],
        "decisiones": "Se optó por clasificación multiclase (3 niveles) en lugar de binaria para mayor granularidad institucional.",
        "entregable": "Documento de alcance, definición de variable objetivo y criterios de éxito.",
    },
    {
        "num": "2", "titulo": "Comprensión de los datos",
        "color": "#3FB950", "bg": "rgba(63,185,80,.07)", "border": "rgba(63,185,80,.25)",
        "objetivo": "Explorar el dataset, detectar calidad de datos y relaciones entre variables.",
        "actividades": [
            "Análisis exploratorio: distribuciones, outliers, valores nulos (~4.1% del total).",
            "Correlaciones con Total_Score: asistencia (0.46), horas de estudio (0.35), estrés (-0.28).",
            "Identificación de data leakage potencial: Midterm_Score, Final_Score son componentes de Total_Score.",
            "Análisis de balance de clases: Riesgo Alto ~5.58%, Riesgo Medio ~81.3%, Riesgo Bajo ~13.1%.",
            "Distribución por departamento: CS, Engineering, Mathematics, Business.",
        ],
        "decisiones": "Se documentó el leakage en lugar de eliminarlo, para demostrar su efecto en métricas vs. modelo conductual.",
        "entregable": "Dashboard EDA interactivo con filtros por cohorte.",
    },
    {
        "num": "3", "titulo": "Preparación de datos",
        "color": "#F59E0B", "bg": "rgba(245,158,11,.07)", "border": "rgba(245,158,11,.25)",
        "objetivo": "Transformar los datos crudos en un formato apto para el modelado.",
        "actividades": [
            "Imputación de nulos: mediana para variables numéricas (robusto ante outliers).",
            "Label Encoding para variables categóricas: Gender, Department, Extracurricular, Internet, Parent Education, Family Income.",
            "Creación de variable objetivo: Riesgo_Academico = {0: Alto, 1: Medio, 2: Bajo} según umbrales de Total_Score.",
            "StandardScaler aplicado exclusivamente a Logistic Regression (RF y DT no requieren escala).",
            "Split estratificado 80/20 con random_state=42 para reproducibilidad.",
        ],
        "decisiones": "Mediana en lugar de media para imputación; preserva robustez ante la distribución no normal de algunas variables.",
        "entregable": "Pipeline de preprocesamiento encapsulado en ModeloRiesgo.preparar().",
    },
    {
        "num": "4", "titulo": "Modelado",
        "color": "#8B5CF6", "bg": "rgba(139,92,246,.07)", "border": "rgba(139,92,246,.25)",
        "objetivo": "Entrenar y comparar 4 algoritmos de clasificación.",
        "actividades": [
            "Logistic Regression: baseline lineal, max_iter=1000, C=1.0, class_weight='balanced'.",
            "Decision Tree: interpretable, max_depth=8, class_weight='balanced'.",
            "Random Forest: modelo principal, n_estimators=200, max_depth=12, class_weight='balanced', n_jobs=-1.",
            "XGBoost: high-performance, n_estimators=200, max_depth=6, learning_rate=0.1.",
            "Validación cruzada k=5 estratificada (StratifiedKFold) para estimar varianza real del desempeño.",
        ],
        "decisiones": "Random Forest como modelo principal por balance entre interpretabilidad (importancia de variables) y rendimiento. class_weight='balanced' compensa el desbalance de clases.",
        "entregable": "4 modelos entrenados, métricas CV-5, importancia de variables, objetos serializables en session_state.",
    },
    {
        "num": "5", "titulo": "Evaluación",
        "color": "#EF4444", "bg": "rgba(239,68,68,.07)", "border": "rgba(239,68,68,.25)",
        "objetivo": "Validar que los modelos cumplen los objetivos del negocio y cuantificar el leakage.",
        "actividades": [
            "Métricas: Accuracy, Precision macro, Recall macro, F1 macro, ROC-AUC (OvR).",
            "Matrices de confusión normalizadas para analizar errores por clase.",
            "Curvas ROC One-vs-Rest para cada modelo.",
            "Modelo conductual entrenado con solo 12 variables conductuales/socioeconómicas para comparativa limpia.",
            "Cuantificación del data leakage: diferencia entre F1 del modelo completo vs. conductual.",
        ],
        "decisiones": "F1 macro como métrica principal por el desbalance de clases; no usar accuracy puro. El modelo conductual sirve como benchmark honesto para el contexto de alerta temprana real.",
        "entregable": "Página Comparativa de Modelos con análisis de leakage, ROC curves y matrices.",
    },
    {
        "num": "6", "titulo": "Comunicación / Despliegue",
        "color": "#10B981", "bg": "rgba(16,185,129,.07)", "border": "rgba(16,185,129,.25)",
        "objetivo": "Presentar resultados y hacer el modelo accesible a usuarios no técnicos.",
        "actividades": [
            "Dashboard EDA interactivo con filtros de cohorte (departamento, género, ingreso, edad).",
            "Módulo de predicción individual con SHAP explainability y simulador de intervención.",
            "Tabla de alertas con filtros, ranking por probabilidad de riesgo y exportación CSV.",
            "Aplicación Streamlit multi-página con design system institucional UDeC.",
            "Documentación metodológica (esta página) accesible desde la interfaz.",
        ],
        "decisiones": "Streamlit elegido por rapidez de prototipado y despliegue. Design system oscuro para coherencia institucional. SHAP para dar explicabilidad a docentes y coordinadores.",
        "entregable": "Aplicación web completa: 6 módulos, sidebar con estado del modelo, exportación de datos.",
    },
]

# Tabs por fase
tab_labels = [f"Fase {f['num']} · {f['titulo']}" for f in fases_info]
tabs = st.tabs(tab_labels)

for tab, fase in zip(tabs, fases_info):
    with tab:
        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.markdown(f"""
            <div style="background:{fase['bg']};border:1px solid {fase['border']};
                        border-left:4px solid {fase['color']};border-radius:10px;
                        padding:1.25rem 1.5rem;margin-bottom:1rem">
              <div style="font-size:.7rem;color:{fase['color']};font-weight:700;
                          text-transform:uppercase;letter-spacing:.08em;margin-bottom:.3rem">
                Fase {fase['num']} · Objetivo
              </div>
              <div style="font-size:.95rem;color:#E6EDF3;font-weight:500">
                {fase['objetivo']}
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"<span style='font-size:.75rem;color:{fase['color']};font-weight:600;text-transform:uppercase;letter-spacing:.06em'>Actividades realizadas</span>", unsafe_allow_html=True)
            for act in fase["actividades"]:
                st.markdown(f"""
                <div style="display:flex;gap:.6rem;align-items:flex-start;
                            padding:.4rem 0;border-bottom:1px solid #21262D">
                  <span style="color:{fase['color']};font-size:1rem;flex-shrink:0;margin-top:.05rem">▸</span>
                  <span style="font-size:.86rem;color:#C9D1D9;line-height:1.5">{act}</span>
                </div>
                """, unsafe_allow_html=True)

        with col_right:
            st.markdown(f"""
            <div style="background:#161B22;border:1px solid #21262D;border-radius:10px;
                        padding:1rem 1.25rem;margin-bottom:1rem">
              <div style="font-size:.7rem;color:#8B949E;font-weight:600;text-transform:uppercase;
                          letter-spacing:.06em;margin-bottom:.5rem">⚖ Decisión técnica clave</div>
              <div style="font-size:.85rem;color:#C9D1D9;line-height:1.6">{fase['decisiones']}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="background:rgba(16,185,129,.07);border:1px solid rgba(16,185,129,.25);
                        border-radius:10px;padding:1rem 1.25rem">
              <div style="font-size:.7rem;color:#3FB950;font-weight:600;text-transform:uppercase;
                          letter-spacing:.06em;margin-bottom:.5rem">Entregable</div>
              <div style="font-size:.85rem;color:#C9D1D9;line-height:1.6">{fase['entregable']}</div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")

# ─────────────────────────────────────────────────────────────────
# Diagrama de flujo CRISP-DM
# ─────────────────────────────────────────────────────────────────
st.markdown("#### Flujo CRISP-DM en este proyecto")

phases_chart = [
    ("1. Comprensión\ndel negocio", "#58A6FF",
     "Objetivo: alerta temprana\nVariable: Riesgo_Academico\n3 clases (Alto/Medio/Bajo)"),
    ("2. Comprensión\nde los datos", "#3FB950",
     "5,000 est. · 23 vars\n4.1% nulos · EDA\nLeakage identificado"),
    ("3. Preparación\nde datos", "#F59E0B",
     "Imputación mediana\nLabel Encoding\nSplit 80/20 estratif."),
    ("4. Modelado", "#8B5CF6",
     "LR · DT · RF · XGBoost\nCV-5 estratificado\nclass_weight balanced"),
    ("5. Evaluación", "#EF4444",
     "F1 macro · ROC-AUC\nLeakage cuantificado\nModelo conductual"),
    ("6. Comunicación", "#10B981",
     "App Streamlit\nDashboard · SHAP\nTabla de alertas"),
]

fig_flow = go.Figure()

def hex_to_rgba(hex_color, alpha=0.08):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

for i, (label, color, desc) in enumerate(phases_chart):
    x = i * 110 + 30
    # Caja principal
    fig_flow.add_shape(type="rect", x0=x, y0=120, x1=x+90, y1=220,
                       fillcolor=hex_to_rgba(color, 0.08), line_color=color, line_width=1.5)
    # Número grande
    fig_flow.add_annotation(x=x+45, y=185, text=f"<b>{i+1}</b>",
                             font=dict(color=color, size=28), showarrow=False)
    # Título
    fig_flow.add_annotation(x=x+45, y=140, text=label.replace("\n", "<br>"),
                             font=dict(color="#E6EDF3", size=10), showarrow=False,
                             align="center")
    # Descripción debajo
    fig_flow.add_annotation(x=x+45, y=90, text=desc.replace("\n", "<br>"),
                             font=dict(color="#8B949E", size=8.5), showarrow=False,
                             align="center")
    # Flecha hacia siguiente
    if i < len(phases_chart) - 1:
        fig_flow.add_annotation(
            x=x+100, y=170, ax=x+92, ay=170,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1.2,
            arrowcolor="#30363D", arrowwidth=1.5,
        )

# Flecha de retro-alimentación (de 6 a 1)
fig_flow.add_shape(type="path",
    path="M 650 220 Q 650 250 340 250 Q 30 250 30 220",
    line=dict(color="#30363D", width=1, dash="dot"))
fig_flow.add_annotation(
    x=340, y=258, text="↺ retroalimentación iterativa",
    font=dict(color="#484F58", size=9), showarrow=False)

fig_flow.update_layout(
    plot_bgcolor="#0D1117", paper_bgcolor="#161B22",
    xaxis=dict(visible=False, range=[-10, 690]),
    yaxis=dict(visible=False, range=[40, 280]),
    margin=dict(l=10, r=10, t=20, b=10),
    height=300,
    font=dict(color="#8B949E"),
)
st.plotly_chart(fig_flow, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# Dataset y variables
# ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### Dataset y variables del proyecto")

col_d1, col_d2, col_d3 = st.columns(3)

with col_d1:
    st.markdown("""
    <div class="risk-card">
      <div style="font-size:.7rem;color:#58A6FF;font-weight:600;text-transform:uppercase;
                  letter-spacing:.06em;margin-bottom:.7rem">Dataset</div>
      <div style="font-size:.85rem;color:#C9D1D9;line-height:2">
        <span style="color:#8B949E">Nombre:</span> Students Performance Dataset<br>
        <span style="color:#8B949E">Fuente:</span> Proveedor privado<br>
        <span style="color:#8B949E">Registros:</span> 5,000 estudiantes<br>
        <span style="color:#8B949E">Atributos:</span> 23 variables<br>
        <span style="color:#8B949E">Nulos:</span> ~4.1% → imputación mediana<br>
        <span style="color:#8B949E">Duplicados:</span> 0<br>
        <span style="color:#8B949E">Formato:</span> CSV codificado UTF-8
      </div>
    </div>
    """, unsafe_allow_html=True)

with col_d2:
    st.markdown("""
    <div class="risk-card">
      <div style="font-size:.7rem;color:#3FB950;font-weight:600;text-transform:uppercase;
                  letter-spacing:.06em;margin-bottom:.7rem">Variable objetivo</div>
      <div style="font-size:.85rem;color:#C9D1D9;line-height:2.2">
        <span style="color:#8B949E">Base:</span> Total_Score (0–100)<br>
        <span style="color:#EF4444">■ Riesgo Alto</span>&nbsp;&nbsp;→ Total &lt; 60<br>
        <span style="color:#F59E0B">■ Riesgo Medio</span> → Total 60–80<br>
        <span style="color:#10B981">■ Riesgo Bajo</span>&nbsp;&nbsp;→ Total &gt; 80<br>
        <span style="color:#8B949E">Codificación:</span> {0: Alto, 1: Medio, 2: Bajo}<br>
        <span style="color:#8B949E">Balance:</span> ~5.58% · ~81.3% · ~13.1%
      </div>
    </div>
    """, unsafe_allow_html=True)

with col_d3:
    st.markdown("""
    <div class="risk-card">
      <div style="font-size:.7rem;color:#F59E0B;font-weight:600;text-transform:uppercase;
                  letter-spacing:.06em;margin-bottom:.7rem">Pipeline técnico</div>
      <div style="font-size:.85rem;color:#C9D1D9;line-height:2">
        <span style="color:#8B949E">Lenguaje:</span> Python 3.10+<br>
        <span style="color:#8B949E">UI:</span> Streamlit 1.x<br>
        <span style="color:#8B949E">ML:</span> scikit-learn · XGBoost<br>
        <span style="color:#8B949E">Visualización:</span> Plotly Express/GO<br>
        <span style="color:#8B949E">Explainability:</span> SHAP (TreeExplainer)<br>
        <span style="color:#8B949E">Split:</span> 80/20 estratificado seed=42<br>
        <span style="color:#8B949E">CV:</span> StratifiedKFold k=5
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Variables table
st.markdown("##### Variables del modelo")

vars_data = [
    ("Attendance (%)",          "Numérica",     "Asistencia del estudiante",                 "Conductual", "Sí"),
    ("Midterm_Score",           "Numérica",     "Nota del examen parcial",                   "Académica ⚠",  "Sí (leakage)"),
    ("Final_Score",             "Numérica",     "Nota del examen final",                     "Académica ⚠",  "Sí (leakage)"),
    ("Assignments_Avg",         "Numérica",     "Promedio de tareas",                        "Académica ⚠",  "Sí (leakage)"),
    ("Quizzes_Avg",             "Numérica",     "Promedio de quizzes",                       "Académica ⚠",  "Sí (leakage)"),
    ("Participation_Score",     "Numérica",     "Puntaje de participación en clase",         "Académica ⚠",  "Sí (leakage)"),
    ("Projects_Score",          "Numérica",     "Nota de proyectos",                         "Académica ⚠",  "Sí (leakage)"),
    ("Study_Hours_per_Week",    "Numérica",     "Horas de estudio por semana",               "Conductual", "Sí"),
    ("Stress_Level (1-10)",     "Numérica",     "Nivel de estrés autoreportado",             "Conductual", "Sí"),
    ("Sleep_Hours_per_Night",   "Numérica",     "Horas de sueño por noche",                  "Conductual", "Sí"),
    ("Age",                     "Numérica",     "Edad del estudiante",                       "Sociodemog.", "Sí"),
    ("Gender",                  "Categórica",   "Género (Male/Female)",                      "Sociodemog.", "Sí (enc.)"),
    ("Department",              "Categórica",   "Programa académico",                        "Sociodemog.", "Sí (enc.)"),
    ("Extracurricular_Activities", "Categórica","Participación extracurricular (Yes/No)",    "Conductual", "Sí (enc.)"),
    ("Internet_Access_at_Home", "Categórica",   "Acceso a internet en casa (Yes/No)",        "Socioeconóm.", "Sí (enc.)"),
    ("Parent_Education_Level",  "Categórica",   "Nivel educativo del padre/madre",           "Socioeconóm.", "Sí (enc.)"),
    ("Family_Income_Level",     "Categórica",   "Nivel de ingreso familiar (Low/Med/High)",  "Socioeconóm.", "Sí (enc.)"),
    ("Total_Score",             "Numérica",     "Score total (BASE de variable objetivo)",   "Objetivo",   "No (es la base)"),
]

import pandas as pd
df_vars = pd.DataFrame(vars_data, columns=["Variable", "Tipo", "Descripción", "Categoría", "En modelo"])

def color_cat(val):
    if "leakage" in val.lower():
        return "color: #EF4444"
    if "enc." in val.lower():
        return "color: #F59E0B"
    if val == "No (es la base)":
        return "color: #8B949E"
    if val == "Sí":
        return "color: #10B981"
    return ""

def color_tipo(val):
    colors = {
        "Académica ⚠": "color: #EF4444",
        "Conductual":  "color: #58A6FF",
        "Socioeconóm.": "color: #8B5CF6",
        "Sociodemog.": "color: #F59E0B",
        "Objetivo":    "color: #8B949E",
    }
    return colors.get(val, "")

st.dataframe(
    df_vars.style
        .map(color_cat, subset=["En modelo"])
        .applymap(color_tipo, subset=["Categoría"]),
    use_container_width=True,
    hide_index=True,
    height=520,
)

st.markdown("""
<div class="info-box" style="margin-top:.75rem">
  ℹ️ &nbsp;<b>Leyenda:</b>
  <span style="color:#10B981">Sí</span> = incluida sin transformación ·
  <span style="color:#F59E0B">Sí (enc.)</span> = incluida tras Label Encoding ·
  <span style="color:#EF4444">Sí (leakage)</span> = incluida pero produce data leakage ·
  <span style="color:#8B949E">No</span> = excluida del modelo
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")

# ─────────────────────────────────────────────────────────────────
# Equipo y contexto
# ─────────────────────────────────────────────────────────────────
st.markdown("#### Información del proyecto")

col_e1, col_e2 = st.columns(2)
with col_e1:
    st.markdown("""
    <div class="risk-card">
      <div style="font-size:.7rem;color:#58A6FF;font-weight:600;text-transform:uppercase;
                  letter-spacing:.06em;margin-bottom:.7rem">Equipo</div>
      <div style="font-size:.9rem;color:#C9D1D9;line-height:2.2">
        <b style="color:#E6EDF3">Juan Miguel Pérez Gil</b><br>
        <span style="color:#8B949E">Estudiante · MD 801</span><br><br>
        <b style="color:#E6EDF3">Carlos David Torrado Estupiñan</b><br>
        <span style="color:#8B949E">Estudiante · MD 801</span><br><br>
        <span style="color:#8B949E">Docente:</span>
        <b style="color:#E6EDF3">Oscar Jobany Gómez Ochoa</b>
      </div>
    </div>
    """, unsafe_allow_html=True)

with col_e2:
    st.markdown("""
    <div class="risk-card">
      <div style="font-size:.7rem;color:#3FB950;font-weight:600;text-transform:uppercase;
                  letter-spacing:.06em;margin-bottom:.7rem">Contexto académico</div>
      <div style="font-size:.9rem;color:#C9D1D9;line-height:2.2">
        <span style="color:#8B949E">Asignatura:</span>
        <b style="color:#E6EDF3">Minería de Datos · MD 801</b><br>
        <span style="color:#8B949E">Institución:</span>
        <b style="color:#E6EDF3">Universidad de Cundinamarca</b><br>
        <span style="color:#8B949E">Año:</span> <b style="color:#E6EDF3">2026</b><br>
        <span style="color:#8B949E">Metodología:</span>
        <b style="color:#E6EDF3">CRISP-DM (6 fases completas)</b><br>
        <span style="color:#8B949E">Tecnología:</span>
        <b style="color:#E6EDF3">Python · Streamlit · scikit-learn · XGBoost</b>
      </div>
    </div>
    """, unsafe_allow_html=True)