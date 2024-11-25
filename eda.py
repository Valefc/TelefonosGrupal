import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px

# Configurar la página de Streamlit
st.set_page_config(layout="wide")
st.title("Análisis Exploratorio de Datos - Visualización de Distribuciones")

# Subir archivo Excel
st.sidebar.header("Sube tu archivo de datos")
uploaded_file = st.sidebar.file_uploader("Selecciona un archivo Excel", type=["xlsx", "xls"])

if uploaded_file is not None:
    # Cargar el archivo en un DataFrame
    df = pd.read_excel(uploaded_file)
    
    # Mostrar el DataFrame cargado
    st.subheader("Vista previa del dataset")
    st.dataframe(df.head())
    
    # Seleccionar columnas de interés
    variables = ['precio_usd', 'ram', 'almacenamiento', 'tamaño_bateria']
    if all(col in df.columns for col in variables):
        st.subheader("Distribución de Variables Cuantitativas")

        # Función para graficar distribuciones
        def plot_distribution(data, column, title):
            plt.figure(figsize=(8, 5))
            sns.histplot(data[column], kde=True, bins=30, color='blue', alpha=0.6)
            plt.title(title)
            plt.xlabel(column)
            plt.ylabel("Frecuencia")
            st.pyplot(plt)  # Mostrar en Streamlit

        # Generar gráficos para cada variable seleccionada
        for var in variables:
            st.write(f"### {var.capitalize()}")
            plot_distribution(df, var, f"Distribución de {var.capitalize()}")
    else:
        st.error("El dataset no contiene las columnas.")
    # Seleccionar las columnas numéricas de interés
    numeric_columns = ['precio_usd', 'almacenamiento', 'peso', 'tamaño_pantalla', 'bateria', 'densidad_ppi']
    if all(col in df.columns for col in numeric_columns):
        # Crear matriz de correlación
        correlation_matrix = df[numeric_columns].corr()

        st.subheader("Matriz de Correlación")
        st.write(correlation_matrix)  # Mostrar la matriz en Streamlit como tabla

        # Visualizar la matriz de correlación con un mapa de calor
        st.subheader("Mapa de Calor de la Matriz de Correlación")
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True, cbar=True)
        plt.title("Matriz de Correlación")
        st.pyplot(plt)  # Mostrar el gráfico en Streamlit
    else:
        st.error("El dataset no contiene las columnas.")

    # Gráficos
    col1, col2 = st.columns((2))  # Crear dos columnas

    with col1:
        st.subheader("Distribución de Marcas de Teléfono")
        fig = px.histogram(
            df,
            x='marca_telefono',
            title="Distribución de las Marcas de Teléfono",
            labels={'marca_telefono': 'Marca de Teléfono', 'count': 'Cantidad de Teléfonos'},
            color='marca_telefono',
            text_auto=True  # Mostrar los valores en las barras
        )
        fig.update_layout(
            xaxis_title="Marca de Teléfono",
            yaxis_title="Cantidad de Teléfonos",
            xaxis={'categoryorder': 'total descending'}  # Ordenar las barras por frecuencia
        )
        st.plotly_chart(fig)

    with col2:
        st.subheader("Distribución de los Sistemas Operativos")
        fig = px.histogram(
            df,
            x='sistema_operativo',
            title="Distribución de los Sistemas Operativos",
            labels={'sistema_operativo': 'Sistema Operativo', 'count': 'Cantidad de Teléfonos'},
            color='sistema_operativo',
            text_auto=True  # Mostrar los valores en las barras
        )
        fig.update_layout(
            xaxis_title="Sistema Operativo",
            yaxis_title="Cantidad de Teléfonos",
            xaxis={'categoryorder': 'total descending'}  # Ordenar las barras por frecuencia
        )
        st.plotly_chart(fig)

    # Para Almacenamiento
    fig = px.bar(
        df.groupby('almacenamiento')['precio_usd'].mean().reset_index(),
        x='almacenamiento',
        y='precio_usd',
        title="Precio Promedio por Almacenamiento",
        labels={'almacenamiento': 'Almacenamiento (GB)', 'precio_usd': 'Precio Promedio (USD)'},
        color='almacenamiento',
        text_auto=True
    )
    st.plotly_chart(fig)

    # Para RAM
    fig = px.bar(
        df.groupby('ram')['precio_usd'].mean().reset_index(),
        x='ram',
        y='precio_usd',
        title="Precio Promedio por RAM",
        labels={'ram': 'RAM (GB)', 'precio_usd': 'Precio Promedio (USD)'},
        color='ram',
        text_auto=True
    )
    st.plotly_chart(fig)

    # Para Batería
    fig = px.bar(
        df.groupby('bateria')['precio_usd'].mean().reset_index(),
        x='bateria',
        y='precio_usd',
        title="Precio Promedio por Batería",
        labels={'bateria': 'Batería (mAh)', 'precio_usd': 'Precio Promedio (USD)'},
        color='bateria',
        text_auto=True
    )
    st.plotly_chart(fig)
else:
    st.warning("Por favor, sube un archivo Excel para comenzar el análisis.")
