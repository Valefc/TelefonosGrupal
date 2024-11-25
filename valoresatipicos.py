import pandas as pd
import matplotlib.pyplot as plt

plt.switch_backend('TkAgg')
# Leer el archivo Excel
ruta_excel = "D:/Documentos/VALERIA/Documents/Univalle 4 Vale/Proyecto Integrador/TelefonosGrupal/TelefonosGrupal/df/Datos_Telefonos_cambios_fin.xlsx" # Cambia esto por la ruta de tu archivo Excel
df = pd.read_excel(ruta_excel)

# Función para detectar y remover outliers usando el método IQR
def remover_outliers(df, columna):
    Q1 = df[columna].quantile(0.25)
    Q3 = df[columna].quantile(0.75)
    IQR = Q3 - Q1
    # Definir los límites para detectar los outliers
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    # Filtrar los datos dentro de los límites
    df_filtrado = df[(df[columna] >= limite_inferior) & (df[columna] <= limite_superior)]
    return df_filtrado

def visualizar_boxplot(df, columnas, nuevo_titulo):
    plt.figure(figsize=(12, 8))
    df[columnas].boxplot()
    plt.title(nuevo_titulo)  # Cambia el título a lo que desees
    plt.show()  # Mostrar la gráfica en pantalla

# Columnas a tratar
columnas_a_tratar = ['precio_usd', 'tamaño_pantalla', 'peso', 'bateria']

# Nuevo título para la gráfica
nuevo_titulo = "Distribución de Variables antes del Tratamiento de Outliers"

# Visualizar datos antes de eliminar outliers con el nuevo título
visualizar_boxplot(df, columnas_a_tratar, nuevo_titulo)

# Remover outliers en las columnas seleccionadas
for columna in columnas_a_tratar:
    df = remover_outliers(df, columna)

# Nuevo título para la gráfica después de eliminar outliers
nuevo_titulo_despues = "Distribución de Variables después del Tratamiento de Outliers"

# Visualizar datos después del tratamiento de outliers con el nuevo título
visualizar_boxplot(df, columnas_a_tratar, nuevo_titulo_despues)

# Guardar el DataFrame limpio en un nuevo archivo Excel
df.to_excel("D:/Documentos/VALERIA/Documents/Univalle 4 Vale/Proyecto Integrador/TelefonosGrupal/TelefonosGrupal/df/datos_sin_outliers_final1.xlsx", index=False)  # Cambia la ruta al archivo de destino
print("Tratamiento de outliers completado y datos guardados 'datos_sin_outliers_final1.xlsx")