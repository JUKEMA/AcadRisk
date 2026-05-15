"""
app.py · Sistema de Alerta Temprana Académica
Streamlit entry point — navegación multi-página con sidebar institucional
"""

import streamlit as st

st.set_page_config(
    page_title="Alerta Temprana Académica · UDeC",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS global institucional
st.markdown("""
<style>
/* ── Tipografía y base */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Sidebar */
[data-testid="stSidebar"] {
    background: #0D1117 !important;
    border-right: 1px solid #21262D;
}
[data-testid="stSidebar"] * { color: #C9D1D9 !important; }
[data-testid="stSidebar"] hr { border-color: #21262D !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #58A6FF !important;
    font-size: 0.75rem !important;
    letter-spacing: .08em;
    text-transform: uppercase;
    font-weight: 600;
}

/* ── Main background */
[data-testid="stAppViewContainer"] > .main {
    background: #0D1117;
}
.block-container {
    padding: 1.5rem 2rem 2rem !important;
    max-width: 1400px;
}

/* ── Métricas */
[data-testid="metric-container"] {
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 10px;
    padding: 1rem 1.25rem !important;
}
[data-testid="metric-container"] label {
    color: #8B949E !important;
    font-size: 0.75rem !important;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: .05em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 1.8rem !important;
    font-weight: 600;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 0.78rem !important;
}

/* ── DataFrames */
[data-testid="stDataFrame"] {
    border: 1px solid #21262D !important;
    border-radius: 8px;
    overflow: hidden;
}

/* ── Tabs */
[data-testid="stTabs"] [role="tablist"] {
    background: #161B22;
    border-radius: 8px 8px 0 0;
    border-bottom: 1px solid #21262D;
    gap: 0;
}
[data-testid="stTabs"] button[role="tab"] {
    color: #8B949E !important;
    font-size: 0.82rem !important;
    font-weight: 500;
    padding: 0.6rem 1.1rem;
    border-radius: 0;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #58A6FF !important;
    border-bottom: 2px solid #58A6FF !important;
    background: transparent !important;
}

/* ── Botones primarios */
.stButton > button[kind="primary"] {
    background: #1F6FEB !important;
    border: none !important;
    color: #fff !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    transition: background .2s;
}
.stButton > button[kind="primary"]:hover {
    background: #388BFD !important;
}

/* ── Inputs y selects */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div {
    background: #161B22 !important;
    border: 1px solid #30363D !important;
    color: #E6EDF3 !important;
    border-radius: 6px !important;
}

/* ── Expander */
details summary {
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 8px;
    padding: .5rem 1rem;
    color: #C9D1D9 !important;
    font-weight: 500;
}

/* ── Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0D1117; }
::-webkit-scrollbar-thumb { background: #30363D; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #484F58; }

/* ── Page title style */
.page-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding-bottom: 1rem;
    border-bottom: 1px solid #21262D;
    margin-bottom: 1.5rem;
}
.page-header h1 {
    font-size: 1.35rem;
    font-weight: 600;
    color: #E6EDF3;
    margin: 0;
}
.page-header .subtitle {
    font-size: 0.8rem;
    color: #8B949E;
    margin: 0;
}
/* ── Cards custom */
.risk-card {
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 10px;
    padding: 1rem 1.25rem;
}
/* ── Alerta banners */
.alert-alto {
    background: rgba(239,68,68,.12);
    border-left: 4px solid #EF4444;
    border-radius: 6px;
    padding: .7rem 1rem;
    color: #FCA5A5;
    font-size: .88rem;
    margin: .5rem 0;
}
.alert-medio {
    background: rgba(245,158,11,.12);
    border-left: 4px solid #F59E0B;
    border-radius: 6px;
    padding: .7rem 1rem;
    color: #FCD34D;
    font-size: .88rem;
    margin: .5rem 0;
}
.alert-bajo {
    background: rgba(16,185,129,.12);
    border-left: 4px solid #10B981;
    border-radius: 6px;
    padding: .7rem 1rem;
    color: #6EE7B7;
    font-size: .88rem;
    margin: .5rem 0;
}
.info-box {
    background: rgba(88,166,255,.08);
    border: 1px solid rgba(88,166,255,.25);
    border-radius: 8px;
    padding: .75rem 1rem;
    color: #79C0FF;
    font-size: .85rem;
    margin: .5rem 0;
}
</style>
""", unsafe_allow_html=True)

# ── Inicializar estado global
if "modelo" not in st.session_state:
    from modelo_riesgo import ModeloRiesgo
    st.session_state.modelo = ModeloRiesgo()
if "df_raw" not in st.session_state:
    st.session_state.df_raw = None

# ── Sidebar logo + info
with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 .5rem">
      <div style="font-size:1.6rem;font-weight:700;color:#58A6FF;letter-spacing:-.01em">
        🎓 AcadRisk
      </div>
      <div style="font-size:.72rem;color:#6E7681;margin-top:.1rem;font-weight:500">
        Sistema de Alerta Temprana Académica
      </div>
      <div style="font-size:.7rem;color:#30363D;margin-top:.3rem">
        MD 801 · UDeC · CRISP-DM · 2026
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown("### Navegación")
    st.page_link("app.py",                    label="🏠  Inicio",             )
    st.page_link("pages/01_entrenamiento.py", label="⚡  Entrenamiento",      )
    st.page_link("pages/02_dashboard.py",     label="📊  Dashboard & EDA",    )
    st.page_link("pages/03_prediccion.py",    label="🎯  Predicción Individual",)
    st.page_link("pages/04_alertas.py",       label="🚨  Tabla de Alertas",   )
    st.page_link("pages/05_modelos.py",       label="🔬  Comparativa Modelos",)
    st.page_link("pages/06_metodologia.py",   label="📖  Metodología CRISP-DM",)

    st.divider()
    modelo = st.session_state.modelo
    if modelo.trained:
        rf = modelo.resultados.get("Random Forest", {})
        st.markdown(f"""
        <div style="background:#0F2A1A;border:1px solid #1A5C35;border-radius:8px;padding:.75rem 1rem">
          <div style="font-size:.7rem;color:#3FB950;font-weight:600;text-transform:uppercase;letter-spacing:.05em">
            ✓ Modelo activo
          </div>
          <div style="font-size:.85rem;color:#7EE2A8;margin:.3rem 0 .1rem">
            Random Forest
          </div>
          <div style="font-size:.75rem;color:#3FB950">
            Accuracy: {rf.get('accuracy','—')}<br>
            F1 macro: {rf.get('f1','—')}<br>
            Train: {modelo.n_train} · Test: {modelo.n_test}
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#1C1107;border:1px solid #4D2D00;border-radius:8px;padding:.75rem 1rem">
          <div style="font-size:.7rem;color:#F0883E;font-weight:600">⚠ Sin modelo entrenado</div>
          <div style="font-size:.75rem;color:#8B6914;margin-top:.3rem">
            Carga el dataset y entrena en la sección ⚡ Entrenamiento.
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <div style="font-size:.68rem;color:#484F58;line-height:1.6">
      <b style="color:#6E7681">Equipo</b><br>
      Juan Miguel Pérez Gil<br>
      Carlos David Torrado Estupiñan<br><br>
      <b style="color:#6E7681">Docente</b><br>
      Oscar Jobany Gómez Ochoa
    </div>
    """, unsafe_allow_html=True)

# ── Página de inicio
st.markdown("""
<div class="page-header">
  <span style="font-size:2rem">🎓</span>
  <div>
    <h1>Sistema de Alerta Temprana Académica</h1>
    <p class="subtitle">Modelo Predictivo de Rendimiento y Riesgo Estudiantil · CRISP-DM · MD 801 · Universidad de Cundinamarca · 2026</p>
  </div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1,1,1])
with col1:
    st.markdown("""
    <div class="risk-card">
      <div style="font-size:.7rem;color:#58A6FF;font-weight:600;text-transform:uppercase;letter-spacing:.06em;margin-bottom:.5rem">
        Flujo de trabajo
      </div>
      <div style="font-size:.88rem;color:#C9D1D9;line-height:2">
        1 · Cargar dataset CSV<br>
        2 · Entrenar los 4 modelos<br>
        3 · Explorar el Dashboard EDA<br>
        4 · Predecir estudiantes individuales<br>
        5 · Revisar tabla de alertas<br>
        6 · Comparar métricas de modelos
      </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="risk-card">
      <div style="font-size:.7rem;color:#3FB950;font-weight:600;text-transform:uppercase;letter-spacing:.06em;margin-bottom:.5rem">
        Dataset principal
      </div>
      <div style="font-size:.88rem;color:#C9D1D9;line-height:2">
        <span style="color:#8B949E">Registros:</span> 5,000 estudiantes<br>
        <span style="color:#8B949E">Atributos:</span> 23 variables<br>
        <span style="color:#8B949E">Nulos aprox.:</span> ~4.1% → imputación<br>
        <span style="color:#8B949E">Fuente:</span> Proveedor privado<br>
        <span style="color:#8B949E">Archivo:</span> Students Performance Dataset.csv
      </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="risk-card">
      <div style="font-size:.7rem;color:#F59E0B;font-weight:600;text-transform:uppercase;letter-spacing:.06em;margin-bottom:.5rem">
        Variable objetivo
      </div>
      <div style="font-size:.88rem;color:#C9D1D9;line-height:2.1">
        <span style="color:#EF4444">■</span> Riesgo Alto &nbsp;&nbsp;→ Total_Score &lt; 60<br>
        <span style="color:#F59E0B">■</span> Riesgo Medio → Total_Score 60–80<br>
        <span style="color:#10B981">■</span> Riesgo Bajo &nbsp;&nbsp;→ Total_Score &gt; 80<br>
        <span style="color:#8B949E">Algoritmo principal:</span> Random Forest<br>
        <span style="color:#8B949E">Métricas:</span> Acc · F1 · AUC-ROC · CV-5
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── CRISP-DM phases visual
phases = [
    ("1", "Comprensión\ndel negocio",    "#58A6FF", "Objetivo institucional: detectar riesgo tempranamente."),
    ("2", "Comprensión\nde los datos",   "#3FB950", "EDA: distribuciones, nulos, correlaciones, sesgos."),
    ("3", "Preparación\nde datos",       "#F59E0B", "Imputación, encoding, escalado, variable objetivo."),
    ("4", "Modelado",                    "#8B5CF6", "LR · DT · Random Forest · XGBoost. CV-5 estratificado."),
    ("5", "Evaluación",                  "#EF4444", "Accuracy · F1 macro · ROC-AUC · Matriz de confusión."),
    ("6", "Comunicación",                "#10B981", "Dashboard interactivo · Predicción · SHAP · Alertas."),
]
cols = st.columns(6)
for col, (num, label, color, desc) in zip(cols, phases):
    with col:
        st.markdown(f"""
        <div style="background:#161B22;border:1px solid #21262D;border-top:3px solid {color};
                    border-radius:8px;padding:.85rem .75rem;text-align:center;height:130px;
                    display:flex;flex-direction:column;justify-content:center;">
          <div style="font-size:1.4rem;font-weight:700;color:{color}">{num}</div>
          <div style="font-size:.78rem;font-weight:600;color:#E6EDF3;margin:.25rem 0;
                      white-space:pre-line;line-height:1.3">{label}</div>
          <div style="font-size:.68rem;color:#8B949E;line-height:1.4">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
  ℹ️ &nbsp;<b>Nota sobre data leakage:</b> Las features académicas directas (Midterm, Final, etc.) 
  comparten información con Total_Score. Esto eleva artificialmente las métricas. 
  La página <b>Comparativa de Modelos</b> incluye un modelo "conductual" entrenado solo con 
  variables socioeconómicas/conductuales para comparación metodológicamente limpia.
</div>
""", unsafe_allow_html=True)
