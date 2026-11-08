import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Configuración visual de la página
st.set_page_config(
    page_title="SENATI - Sistema Inteligente Académico",
    page_icon="🎓",
    layout="wide"
)

# -------------------------------------------------------------
# 1. GENERACIÓN / CARGA DE DATOS EN CACHÉ
# -------------------------------------------------------------
@st.cache_data
def generar_datos():
    np.random.seed(42)
    n = 100
    horas = np.random.randint(2, 20, size=n)
    asist = np.random.randint(50, 101, size=n)
    calif = np.clip(
        np.round((horas * 0.45) + (asist * 0.12) + np.random.normal(0, 1.8, n)),
        0, 20
    )
    res = np.where((calif >= 11) & (asist >= 70), 1, 0)
    return pd.DataFrame({
        'horas_estudio': horas,
        'asistencia': asist,
        'calificacion': calif,
        'resultado': res
    })

df = generar_datos()

# -------------------------------------------------------------
# ENCABEZADO
# -------------------------------------------------------------
st.title("🎓 SENATI: Sistema Inteligente de Rendimiento Académico")
st.markdown("Aplicación web de **Inteligencia Artificial** que combina **Aprendizaje Supervisado** (clasificación) y **No Supervisado** (segmentación por clustering).")

# Pestañas principales
tab_supervisado, tab_no_supervisado, tab_datos = st.tabs([
    "🤖 Parte A: Predicción (Supervisado)",
    "🔍 Parte B: Segmentación (No Supervisado)",
    "📋 Ver Conjunto de Datos"
])

# =============================================================
# PESTAÑA 1: APRENDIZAJE SUPERVISADO
# =============================================================
with tab_supervisado:
    st.header("Predicción de Aprobación de Estudiantes")
    st.write("Ingresa los datos de un alumno para predecir si **aprobará o desaprobará** el semestre.")

    col_input, col_resultado = st.columns([1, 1])

    with col_input:
        st.subheader("📝 Datos del Alumno a Evaluar")
        horas_in = st.slider("Horas de estudio semanales:", min_value=1, max_value=25, value=12)
        asist_in = st.slider("Porcentaje de asistencia (%):", min_value=40, max_value=100, value=85)
        nota_in = st.slider("Calificación promedio (0 a 20):", min_value=0, max_value=20, value=13)
        
        algoritmo = st.radio("Selecciona el algoritmo:", ["Random Forest", "K-Nearest Neighbors (KNN)"])
        btn_predecir = st.button("🔮 Evaluar Estudiante", use_container_width=True)

    # Entrenamiento interno de modelos
    X = df[['horas_estudio', 'asistencia', 'calificacion']]
    y = df['resultado']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # Modelos
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    knn_model = KNeighborsClassifier(n_neighbors=3).fit(X_train_sc, y_train)

    with col_resultado:
        st.subheader("📊 Resultado del Modelo")
        if btn_predecir:
            nuevo_alumno = pd.DataFrame([[horas_in, asist_in, nota_in]], columns=X.columns)
            
            if algoritmo == "Random Forest":
                pred = rf_model.predict(nuevo_alumno)[0]
                prob = rf_model.predict_proba(nuevo_alumno)[0][1]
                acc = accuracy_score(y_test, rf_model.predict(X_test))
            else:
                pred = knn_model.predict(scaler.transform(nuevo_alumno))[0]
                prob = knn_model.predict_proba(scaler.transform(nuevo_alumno))[0][1]
                acc = accuracy_score(y_test, knn_model.predict(X_test_sc))

            if pred == 1:
                st.success(f"### 🎉 Resultado: EL ALUMNO APROBARÁ")
            else:
                st.error(f"### ⚠️ Resultado: EL ALUMNO DESAPROBARÁ")

            col_m1, col_m2 = st.columns(2)
            col_m1.metric("Probabilidad de Aprobación", f"{prob*100:.1f}%")
            col_m2.metric("Exactitud del Modelo", f"{acc*100:.1f}%")

    st.divider()
    st.subheader("📈 Matriz de Confusión del Modelo")
    
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        fig1, ax1 = plt.subplots(figsize=(4, 3))
        ConfusionMatrixDisplay.from_predictions(y_test, rf_model.predict(X_test), ax=ax1, cmap='Greens', display_labels=['Desaprueba', 'Aprueba'])
        ax1.set_title("Matriz: Random Forest")
        ax1.grid(False)
        st.pyplot(fig1)

    with col_g2:
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        ConfusionMatrixDisplay.from_predictions(y_test, knn_model.predict(X_test_sc), ax=ax2, cmap='Blues', display_labels=['Desaprueba', 'Aprueba'])
        ax2.set_title("Matriz: KNN")
        ax2.grid(False)
        st.pyplot(fig2)

# =============================================================
# PESTAÑA 2: APRENDIZAJE NO SUPERVISADO
# =============================================================
with tab_no_supervisado:
    st.header("Agrupación de Estudiantes mediante K-Means")
    st.write("Identificación automática de **3 grupos de estudiantes** sin usar la variable de aprobación.")

    # Preparar datos sin etiqueta
    X_clust = df[['horas_estudio', 'asistencia', 'calificacion']]
    scaler_km = StandardScaler()
    X_clust_sc = scaler_km.fit_transform(X_clust)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_clust_sc)

    # Identificación y ordenamiento de grupos
    resumen = df.groupby('cluster')[['calificacion', 'asistencia', 'horas_estudio']].mean().round(2)
    orden = resumen['calificacion'].sort_values().index.tolist()
    
    nombres = {
        orden[0]: "Grupo Crítico (En Riesgo)",
        orden[1]: "Grupo Rendimiento Regular",
        orden[2]: "Grupo Sobresaliente"
    }
    df['perfil'] = df['cluster'].map(nombres)

    col_t1, col_t2 = st.columns([1, 1.3])

    with col_t1:
        st.subheader("📌 Características Promedio por Grupo")
        resumen['Perfil Académico'] = [nombres[i] for i in resumen.index]
        st.dataframe(resumen[['Perfil Académico', 'calificacion', 'asistencia', 'horas_estudio']], use_container_width=True)

        st.info("""
        * **Grupo Crítico:** Alumnos con bajas calificaciones (<11) y baja asistencia. Requieren tutoría.
        * **Grupo Regular:** Asistencia aceptable y notas intermedias (11-14).
        * **Grupo Sobresaliente:** Alta dedicación horaria (>12 hrs) y notas destacadas (>15).
        """)

    with col_t2:
        st.subheader("📊 Gráfico de Agrupación (K-Means)")
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        colores = {
            "Grupo Crítico (En Riesgo)": "#e74c3c",
            "Grupo Rendimiento Regular": "#f39c12",
            "Grupo Sobresaliente": "#27ae60"
        }
        for p, color in colores.items():
            sub = df[df['perfil'] == p]
            ax3.scatter(sub['horas_estudio'], sub['calificacion'], c=color, label=p, s=70, alpha=0.85, edgecolors='black')
        
        ax3.set_xlabel("Horas de Estudio Semanales")
        ax3.set_ylabel("Calificación (0 a 20)")
        ax3.axhline(10.5, color='gray', linestyle='--', label='Aprobación (10.5)')
        ax3.legend(fontsize=8, loc='lower right')
        st.pyplot(fig3)

# =============================================================
# PESTAÑA 3: CONJUNTO DE DATOS
# =============================================================
with tab_datos:
    st.header("📋 Registros de Estudiantes")
    st.write(f"Total de registros cargados: **{len(df)} estudiantes**")
    st.dataframe(df[['horas_estudio', 'asistencia', 'calificacion', 'resultado', 'perfil']], use_container_width=True)