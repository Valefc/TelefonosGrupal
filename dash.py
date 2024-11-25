import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(page_title='Dashboard', page_icon=":bar_chart", layout="wide")

st.title('Dashboard telefonos :cellphone: ')

st.markdown(
    """
        <style>
            .stMetric{
                background-color: #121212;  /* Fondo oscuro para un look tecnológico */
                color: #E0E0E0;             /* Texto en color claro */
                border: 1px solid #333333; /* Borde más delgado y oscuro */
                padding: 15px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
        </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
        <style>
            h1 {
                background-color: #00BFFF; /* Azul brillante y futurista */
                color: #FFFFFF;
                padding: 15px;
                border-radius: 10px;
                text-align: center;
                margin-bottom: 20px;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            }
            h2, h3, h4, h5, h6 {
                background-color: #89CFF0;  /* Gris oscuro, moderno */
                color: #FFFFFF;
                padding: 12px;
                border-radius: 8px;
                text-align: center;
                margin-bottom: 15px;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                box-shadow: 0 2px 5px rgba(0,0,0,0.2);
             }
            [data-testid="stFullScreenFrame"]{
                background-color: silver;
                border-top: 3px;
                border-bottom: 3px;
                border-left: 3px;
                border-radius: 15px 0px 0px 15px;
                padding: 5px;
                
            }
            
            /* Ajuste de los gráficos */
            .plotly-graph-div {
                background-color: #181818; /* Fondo oscuro para gráficos */
                border: 1px solid #444444; /* Borde gris y delgado */
                border-radius: 10px;
            }
        </style>
    """,
    unsafe_allow_html=True
)


# Cargar datos (reemplaza con la ruta de tu archivo)
df = pd.read_excel("D:/Documentos/VALERIA/Documents/Univalle 4 Vale/Proyecto Integrador/TelefonosGrupal/TelefonosGrupal/df/datos_sin_outliers.xlsx")

# Mostrar los datos en un expander

with st.expander("Datos del Dataset"):
    st.write(df)

# Filtros y selecciones (Combos)
st.sidebar.header("Filtros")
marca_select = st.sidebar.selectbox('Selecciona la marca', df['marca_telefono'].unique())
df_filtered = df[df['marca_telefono'] == marca_select]

# Mostrar los datos filtrados
st.subheader(f"Datos filtrados para {marca_select}")
st.write(df_filtered)

# Gráficos interactivos

# a. Scatter plot de precio_usd vs ram
st.subheader("Gráfico de Dispersión: Precio vs RAM")
fig_scatter_ram = px.scatter(df_filtered, x="ram", y="precio_usd", color="marca_telefono", 
                             size="precio_usd", hover_name="modelo_telefono", 
                             title="Precio vs RAM")
st.plotly_chart(fig_scatter_ram)

# b. Scatter plot de precio_usd vs almacenamiento
st.subheader("Gráfico de Dispersión: Precio vs Almacenamiento")
fig_scatter_almacenamiento = px.scatter(df_filtered, x="almacenamiento", y="precio_usd", color="marca_telefono", 
                                        size="precio_usd", hover_name="modelo_telefono", 
                                        title="Precio vs Almacenamiento")
st.plotly_chart(fig_scatter_almacenamiento)

# c. Gráfico de barras para la popularidad de sistemas operativos y tipos de pantalla

# Calcular la frecuencia de cada categoría en 'sistema_operativo'
os_freq = df_filtered['sistema_operativo'].value_counts().reset_index()
os_freq.columns = ['sistema_operativo', 'frecuencia']

# Gráfico de barras de sistemas operativos con frecuencia total
st.subheader("Gráfico de Barras: Popularidad de Sistemas Operativos")
fig_os = px.bar(os_freq, x="sistema_operativo", y="frecuencia", 
                title="Popularidad de Sistemas Operativos", 
                color="sistema_operativo", 
                labels={"frecuencia": "Frecuencia", "sistema_operativo": "Sistema Operativo"})
st.plotly_chart(fig_os)

# Calcular la frecuencia de cada categoría en 'tipo_pantalla'
screen_freq = df_filtered['tipo_pantalla'].value_counts().reset_index()
screen_freq.columns = ['tipo_pantalla', 'frecuencia']

# Gráfico de barras de tipos de pantalla con frecuencia total
st.subheader("Gráfico de Barras: Popularidad de Tipos de Pantalla")
fig_screen = px.bar(screen_freq, x="tipo_pantalla", y="frecuencia", 
                    title="Popularidad de Tipos de Pantalla", 
                    color="tipo_pantalla", 
                    labels={"frecuencia": "Frecuencia", "tipo_pantalla": "Tipo de Pantalla"})
st.plotly_chart(fig_screen)

# d. Heatmap de correlación entre variables específicas
st.subheader("Heatmap de Correlación entre Variables Específicas")
# Selección de variables relevantes
variables = ['almacenamiento', 'ram', 'bateria', 'peso', 'precio_usd', 'tamaño_pantalla', 'densidad_ppi']
df_corr = df[variables].corr()  # Calculamos la correlación de las variables seleccionadas

# Heatmap usando Seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(df_corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
st.pyplot(plt)