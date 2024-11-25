import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.tree import export_graphviz
import graphviz
from sklearn.preprocessing import label_binarize

# Configuración de Streamlit
st.set_page_config(page_title="Random Forest - Clasificación de Precio USD")

# Cargar el dataset
dataset_path = "D:/Documentos/VALERIA/Documents/Univalle 4 Vale/Proyecto Integrador/TelefonosGrupal/TelefonosGrupal/df/datos_sin_outliers.xlsx"
try:
    dataset = pd.read_excel(dataset_path)
except Exception as e:
    st.error(f"No se pudo cargar el archivo: {e}")
    st.stop()

# Análisis de los datos
st.title("Análisis de Datos de Precios de móviles")
with st.expander("Descripción del dataset:"):
    st.write(dataset.describe())

# Validar que las columnas existan en el dataset
required_columns = ['almacenamiento', 'ram', 'tipo_pantalla', 'densidad_ppi', 'precio_usd', 'rango_precio']
missing_columns = [col for col in required_columns if col not in dataset.columns]
if missing_columns:
    st.error(f"Faltan columnas requeridas en el dataset: {', '.join(missing_columns)}")
    st.stop()

# Preprocesamiento de datos
# Convertir tipo_pantalla a numérica
label_encoder = LabelEncoder()
dataset['tipo_pantalla'] = label_encoder.fit_transform(dataset['tipo_pantalla'])

# Variables predictoras y objetivo
X = dataset[['almacenamiento', 'ram', 'tipo_pantalla', 'densidad_ppi']].values
y = dataset['rango_precio']  # Usamos la columna 'rango_precio' como objetivo

# División del dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Entrenamiento del modelo
classifier = RandomForestClassifier(n_estimators=100, random_state=0)
classifier.fit(X_train, y_train)

# Predicciones y probabilidades
y_pred = classifier.predict(X_test)
y_proba = classifier.predict_proba(X_test)

# Interfaz para introducir valores de usuario
st.sidebar.header("Introducir Datos de Usuario")
almacenamiento = st.sidebar.number_input("Almacenamiento (GB)", min_value=32, max_value=1024, value=128, step=32)
ram = st.sidebar.number_input("RAM (GB)", min_value=1, max_value=64, value=8, step=1)
tipo_pantalla = st.sidebar.selectbox("Tipo de Pantalla", options=label_encoder.classes_)
tipo_pantalla_encoded = label_encoder.transform([tipo_pantalla])[0]  # Convertir a numérico
densidad_ppi = st.sidebar.number_input("Densidad PPI", min_value=100, max_value=600, value=300, step=10)

user_input = np.array([[almacenamiento, ram, tipo_pantalla_encoded, densidad_ppi]])

# Botón para realizar la predicción
if st.sidebar.button("Predecir"):
    prediction = classifier.predict(user_input)
    st.sidebar.subheader("Predicción para el Usuario")
    st.sidebar.write(f"Almacenamiento: {almacenamiento} GB, RAM: {ram} GB, Tipo de Pantalla: {tipo_pantalla}, Densidad PPI: {densidad_ppi}")
    st.sidebar.write(f"Predicción del rango de precio: {'Bajo' if prediction[0] == 1 else 'Medio' if prediction[0] == 2 else 'Alto'}")

# Matriz de Confusión
st.title("Matriz de Confusión")
labels = ['Bajo', 'Medio', 'Alto']
conf_matrix = confusion_matrix(y_test, y_pred, labels=[1, 2, 3])
fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicción")
plt.ylabel("Real")
st.pyplot(fig)

# Reporte de Clasificación
st.title("Reporte de Clasificación")
classification_rep = classification_report(y_test, y_pred, target_names=labels, output_dict=True)
classification_df = pd.DataFrame(classification_rep).transpose()
st.dataframe(classification_df)

# Curva ROC
st.title("Curva ROC")
fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1], pos_label=2)  # Aquí tomamos la clase 'Medio' (2)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='blue', label='Curva ROC (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC')
plt.legend(loc='lower right')
st.pyplot(plt)

# Binarizar etiquetas
y_test_binarized = label_binarize(y_test, classes=[1, 2, 3])

# Curva de Precisión-Recall por clase
st.title("Curva de Precisión-Recall por Clase")
for i, class_name in enumerate(['Bajo', 'Medio', 'Alto']):
    precision, recall, _ = precision_recall_curve(y_test_binarized[:, i], y_proba[:, i])
    
    plt.figure()
    plt.plot(recall, precision, color='blue', label=f'Clase: {class_name}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Curva de Precisión-Recall (Clase: {class_name})')
    plt.grid()
    plt.legend()
    st.pyplot(plt)

# Visualización de un Árbol de Decisión en el Random Forest
st.title("Visualización de un Árbol de Decisión en el Random Forest")
tree_index = st.number_input("Índice del árbol", min_value=0, max_value=len(classifier.estimators_)-1, value=0)

dot_data = export_graphviz(
    classifier.estimators_[tree_index],
    out_file=None,
    feature_names=['Almacenamiento', 'RAM', 'Tipo Pantalla', 'Densidad PPI'],
    class_names=labels,
    filled=True, rounded=True,
    special_characters=True
)
graph = graphviz.Source(dot_data)
st.graphviz_chart(graph)

# Validación Cruzada
st.title("Validación Cruzada")
scores = cross_val_score(classifier, X, y, cv=5)
st.write(f"Scores de la validación cruzada: {scores}")
st.write(f"Media de los scores: {round(np.mean(scores), 2)}")
st.write(f"Desviación estándar de los scores: {round(np.std(scores), 2)}")

