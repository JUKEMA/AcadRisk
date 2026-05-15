"""
modelo_riesgo.py
Sistema de Alerta Temprana Académica · MD 801 · UDeC 2026
Encapsula preprocesamiento, entrenamiento, evaluación y predicción.
Metodología CRISP-DM — Fases 3 y 4
"""

import math
import numpy as np
import pandas as pd

FEATURE_COLS = [
    "Attendance (%)", "Midterm_Score", "Final_Score",
    "Assignments_Avg", "Quizzes_Avg", "Participation_Score",
    "Projects_Score", "Study_Hours_per_Week",
    "Stress_Level (1-10)", "Sleep_Hours_per_Night", "Age",
    "Gender_enc", "Dept_enc", "Extracurr_enc",
    "Internet_enc", "ParentEdu_enc", "FamilyInc_enc",
]

BEHAVIORAL_COLS = [
    "Attendance (%)", "Study_Hours_per_Week",
    "Stress_Level (1-10)", "Sleep_Hours_per_Night",
    "Participation_Score", "Extracurr_enc",
    "Internet_enc", "ParentEdu_enc", "FamilyInc_enc", "Age",
    "Gender_enc", "Dept_enc",
]

RIESGO_LABELS  = {0: "Riesgo Alto", 1: "Riesgo Medio", 2: "Riesgo Bajo"}
RIESGO_COLORS  = {0: "#EF4444", 1: "#F59E0B", 2: "#10B981"}

ENC_MAPS = {
    "Gender":                    {"Male": 0, "Female": 1},
    "Department":                {"CS": 0, "Engineering": 1, "Mathematics": 2, "Business": 3},
    "Extracurricular_Activities":{"No": 0, "Yes": 1},
    "Internet_Access_at_Home":   {"No": 0, "Yes": 1},
    "Parent_Education_Level":    {"High School": 0, "Bachelor": 1, "Master": 2, "PhD": 3},
    "Family_Income_Level":       {"Low": 0, "Medium": 1, "High": 2},
}

COL_ENC_NAMES = {
    "Gender": "Gender_enc", "Department": "Dept_enc",
    "Extracurricular_Activities": "Extracurr_enc",
    "Internet_Access_at_Home": "Internet_enc",
    "Parent_Education_Level": "ParentEdu_enc",
    "Family_Income_Level": "FamilyInc_enc",
}

FEATURE_DISPLAY = {
    "Attendance (%)":        "Asistencia (%)",
    "Midterm_Score":         "Nota Parcial",
    "Final_Score":           "Nota Final",
    "Assignments_Avg":       "Promedio Tareas",
    "Quizzes_Avg":           "Promedio Quizzes",
    "Participation_Score":   "Participación",
    "Projects_Score":        "Nota Proyectos",
    "Study_Hours_per_Week":  "Horas Estudio/Sem",
    "Stress_Level (1-10)":   "Nivel Estrés",
    "Sleep_Hours_per_Night": "Horas Sueño",
    "Age":                   "Edad",
    "Gender_enc":            "Género",
    "Dept_enc":              "Departamento",
    "Extracurr_enc":         "Actividades Extra.",
    "Internet_enc":          "Internet en Casa",
    "ParentEdu_enc":         "Educación Padres",
    "FamilyInc_enc":         "Ingreso Familiar",
}


class ModeloRiesgo:
    """
    Encapsula CRISP-DM Fases 3–5.
    Modelo principal: Random Forest.
    Comparativos: Logistic Regression, Decision Tree, XGBoost.
    Variable objetivo: Riesgo_Academico (0=Alto, 1=Medio, 2=Bajo)
    """

    def __init__(self):
        self.trained      = False
        self.modelos      = {}
        self.resultados   = {}
        self.importancia  = {}
        self.df_stats     = {}
        self.df_prepared  = None
        self.scaler       = None
        self.feature_names= []
        self.n_train      = 0
        self.n_test       = 0
        self.cv_scores    = {}

    # ── CRISP-DM Fase 3: Preparación
    def preparar(self, df_raw):
        df = df_raw.copy()

        # Variable objetivo
        def cat_riesgo(s):
            if s > 80:   return 2
            elif s >= 60: return 1
            else:         return 0

        df["Riesgo_Academico"] = df["Total_Score"].apply(cat_riesgo)

        # Encoding categóricas
        for col, enc_col in COL_ENC_NAMES.items():
            mapping = ENC_MAPS[col]
            df[enc_col] = df[col].map(mapping).fillna(0).astype(int)

        # Imputar nulos con mediana
        num_cols = [c for c in FEATURE_COLS if c in df.columns]
        for col in num_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

        # Estadísticas EDA
        self.df_stats = self._calc_stats(df)
        self.df_prepared = df.copy()

        X = df[num_cols].values
        y = df["Riesgo_Academico"].values
        return X, y, num_cols

    def _calc_stats(self, df):
        corr_cols = [
            "Attendance (%)", "Study_Hours_per_Week", "Stress_Level (1-10)",
            "Sleep_Hours_per_Night", "Projects_Score",
            "Midterm_Score", "Total_Score",
        ]
        corr = {}
        sub = df[[c for c in corr_cols if c in df.columns]].dropna()
        for c in sub.columns:
            if c != "Total_Score":
                try:
                    vals_c = sub[c].values
                    vals_t = sub["Total_Score"].values
                    n = len(vals_c)
                    mx = vals_c.mean(); mt = vals_t.mean()
                    num = ((vals_c - mx) * (vals_t - mt)).sum()
                    dx  = np.sqrt(((vals_c - mx)**2).sum() or 1)
                    dt  = np.sqrt(((vals_t - mt)**2).sum() or 1)
                    corr[c] = round(float(num / (dx * dt)), 3)
                except Exception:
                    corr[c] = 0.0

        return {
            "n_total":          len(df),
            "n_riesgo_alto":    int((df["Riesgo_Academico"] == 0).sum()),
            "n_riesgo_medio":   int((df["Riesgo_Academico"] == 1).sum()),
            "n_riesgo_bajo":    int((df["Riesgo_Academico"] == 2).sum()),
            "pct_riesgo_alto":  round(100 * (df["Riesgo_Academico"] == 0).mean(), 1),
            "pct_riesgo_medio": round(100 * (df["Riesgo_Academico"] == 1).mean(), 1),
            "pct_riesgo_bajo":  round(100 * (df["Riesgo_Academico"] == 2).mean(), 1),
            "avg_total":        round(df["Total_Score"].mean(), 2),
            "std_total":        round(df["Total_Score"].std(), 2),
            "avg_asistencia":   round(df["Attendance (%)"].mean(), 2),
            "avg_estudio":      round(df["Study_Hours_per_Week"].mean(), 2),
            "avg_estres":       round(df["Stress_Level (1-10)"].mean(), 2),
            "avg_sueno":        round(df["Sleep_Hours_per_Night"].mean(), 2),
            "dept_counts":      df["Department"].value_counts().to_dict(),
            "gender_counts":    df["Gender"].value_counts().to_dict(),
            "income_counts":    df["Family_Income_Level"].value_counts().to_dict(),
            "correlaciones":    corr,
            "n_nulls":          int(df.isnull().sum().sum()),
        }

    # ── CRISP-DM Fase 4: Modelado
    def entrenar(self, df_raw, progress_cb=None):
        from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import (accuracy_score, precision_score,
                                     recall_score, f1_score, confusion_matrix,
                                     roc_auc_score)

        try:
            from xgboost import XGBClassifier
            tiene_xgb = True
        except ImportError:
            tiene_xgb = False

        if progress_cb: progress_cb(5, "Preparando datos...")
        X, y, feature_names = self.preparar(df_raw)
        self.feature_names = feature_names

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        self.n_train = len(X_train)
        self.n_test  = len(X_test)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)
        self.scaler = scaler

        modelos_def = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000, random_state=42, C=1.0, class_weight="balanced"),
            "Decision Tree": DecisionTreeClassifier(
                max_depth=8, random_state=42, class_weight="balanced"),
            "Random Forest": RandomForestClassifier(
                n_estimators=200, max_depth=12, random_state=42,
                n_jobs=-1, class_weight="balanced"),
        }
        if tiene_xgb:
            modelos_def["XGBoost"] = XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                eval_metric="mlogloss", random_state=42, verbosity=0)

        resultados = {}
        total = len(modelos_def)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for idx, (nombre, modelo) in enumerate(modelos_def.items()):
            pct = 10 + int(70 * idx / total)
            if progress_cb: progress_cb(pct, f"Entrenando {nombre}...")

            if nombre == "Logistic Regression":
                modelo.fit(X_train_s, y_train)
                y_pred  = modelo.predict(X_test_s)
                y_proba = modelo.predict_proba(X_test_s)
                cv_sc   = cross_val_score(modelo, scaler.transform(X), y, cv=cv, scoring="f1_macro")
            else:
                modelo.fit(X_train, y_train)
                y_pred  = modelo.predict(X_test)
                y_proba = modelo.predict_proba(X_test)
                cv_sc   = cross_val_score(modelo, X, y, cv=cv, scoring="f1_macro")

            cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
            try:
                roc_auc = round(float(roc_auc_score(
                    y_test, y_proba, multi_class="ovr", average="macro")), 4)
            except Exception:
                roc_auc = None

            resultados[nombre] = {
                "accuracy":  round(accuracy_score(y_test, y_pred), 4),
                "precision": round(precision_score(y_test, y_pred, average="macro", zero_division=0), 4),
                "recall":    round(recall_score(y_test, y_pred, average="macro", zero_division=0), 4),
                "f1":        round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4),
                "roc_auc":   roc_auc,
                "cm":        cm.tolist(),
                "cv_mean":   round(float(cv_sc.mean()), 4),
                "cv_std":    round(float(cv_sc.std()), 4),
                "y_test":    y_test.tolist(),
                "y_proba":   y_proba.tolist(),
                "modelo":    modelo,
            }

        if progress_cb: progress_cb(85, "Calculando importancia de variables...")
        rf = resultados["Random Forest"]["modelo"]
        imp = rf.feature_importances_
        self.importancia = dict(sorted(
            {fn: round(float(v), 4) for fn, v in zip(feature_names, imp)}.items(),
            key=lambda x: x[1], reverse=True
        ))

        self.modelos = {k: v["modelo"] for k, v in resultados.items()}
        self.resultados = {k: {kk: vv for kk, vv in v.items() if kk != "modelo"}
                           for k, v in resultados.items()}
        self.trained = True

        if progress_cb: progress_cb(100, "¡Entrenamiento completado!")
        return self.resultados

    # ── Predicción individual
    def predecir(self, valores_dict, modelo_nombre="Random Forest"):
        if not self.trained:
            raise RuntimeError("Modelo no entrenado")
        x = np.array([float(valores_dict.get(f, 0.0)) for f in self.feature_names]).reshape(1, -1)
        modelo = self.modelos[modelo_nombre]
        if modelo_nombre == "Logistic Regression":
            x = self.scaler.transform(x)
        pred  = int(modelo.predict(x)[0])
        probs = modelo.predict_proba(x)[0].tolist()
        return pred, probs

    # ── Predicción sobre el dataset completo (para tabla de alerta)
    def predecir_dataset(self, modelo_nombre="Random Forest"):
        if not self.trained or self.df_prepared is None:
            return None
        df = self.df_prepared.copy()
        X  = df[self.feature_names].values
        modelo = self.modelos[modelo_nombre]
        if modelo_nombre == "Logistic Regression":
            X = self.scaler.transform(X)
        preds  = modelo.predict(X)
        probas = modelo.predict_proba(X)
        df["Riesgo_Pred"]    = preds
        df["Riesgo_Label"]   = [RIESGO_LABELS[p] for p in preds]
        df["P_Riesgo_Alto"]  = [round(p[0] * 100, 1) for p in probas]
        df["P_Riesgo_Medio"] = [round(p[1] * 100, 1) for p in probas]
        df["P_Riesgo_Bajo"]  = [round(p[2] * 100, 1) for p in probas]
        df["Confianza (%)"]  = [round(max(p) * 100, 1) for p in probas]
        return df
