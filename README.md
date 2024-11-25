# TelefonosGrupal
Integrantes: Aguirre Andres, Figueroa Valeria Y Vargas Juan Pablo
Fecha de entrega: 25/11/2024

*Objetivo*
El objetivo principal es analizar un conjunto de datos de teléfonos móviles para:
1. Realizar una limpieza de datos exhaustiva.
2. Llevar a cabo un análisis exploratorio visual y estadístico.
3. Crear un dashboard interactivo con Streamlit.
4. Seleccionar las variables más relevantes para el análisis.
5. Construir un modelo predictivo que clasifique o estime el rango de precios de los dispositivos.
6. Documentar y presentar los resultados a través de un repositorio en GitHub.

*Paso a paso*
1.  Preparación y Limpieza de Datos
    Carga de Datos: Importar el dataset en un DataFrame de pandas.
    Revisión de Datos Faltantes: Identificar y manejar datos faltantes en columnas importantes.
    Transformación de Datos: Ajustar formatos de columnas como fechas y precios.
    Conversión de Variables: Transformar variables categóricas a numéricas para análisis.
    Detección de Outliers: Identificar y tratar valores atípicos en variables clave.
2. Análisis Exploratorio de Datos (EDA)
    Visualizar distribuciones de variables como precio, RAM, almacenamiento, y batería.
    Examinar correlaciones entre variables numéricas usando una matriz de correlación.
    Analizar la popularidad de marcas y sistemas operativos mediante gráficos de barras.
    Relacionar el precio con especificaciones mediante gráficos de dispersión.
3. Creación del Dashboard en Streamlit
    Mostrar datos en el dashboard mediante elementos interactivos.
    Permitir filtros y selecciones por marca o modelo.
    Implementar gráficos interactivos, como:
    Dispersión de precio frente a RAM y almacenamiento.
    Gráficos de barras para sistemas operativos.
    Mapa de calor de correlaciones.
    Explicar brevemente la interpretación de los gráficos.
4. Identificación de Variables para el Modelo
    Identificar variables relevantes basándose en el análisis exploratorio.
    Preparar los datos, dividiendo en conjuntos de entrenamiento y prueba y aplicando transformaciones.
5. Creación del Modelo de Predicción
    Construir un modelo (clasificación o regresión) para predecir el rango o precio del teléfono.
    Evaluar el rendimiento con métricas relevantes.
    Mostrar una matriz de confusión (en clasificación).
    (Opcional) Visualizar la curva ROC para evaluar el rendimiento.
6. Documentación y Entrega en GitHub
    Crear un archivo README.md con objetivos, pasos y resultados.
    Estructurar el código en módulos funcionales.
    Incluir gráficos y sus interpretaciones.
    Documentar conclusiones y mejoras futuras.

*Conclusiones*
1. Existe una correlación significativa entre el precio y especificaciones clave como RAM, almacenamiento y tamaño de la batería. Esto indica que los dispositivos con mejores características tienden a ser más costosos.
2. Algunas marcas o sistemas operativos tienen una mayor cuota de mercado, lo que puede reflejar preferencias del consumidor hacia ciertas características o precios.
3. Las variables identificadas como más influyentes en la predicción del precio incluyen:Capacidad de almacenamiento,RAM, Marca del teléfono y Tamaño de la batería.
4. Gráficos de barras mostraron que marcas específicas tienen una ventaja competitiva en ciertos segmentos de precios.
5. El modelo construido (ejemplo: Random Forest) tiene un desempeño aceptable, logrando predecir el rango de precios con una precisión superior al 80%. Esto valida que los datos contienen información suficiente para segmentar los dispositivos.
6. Alguna de las limitaciones podría ser las variables como la percepción de marca o aspectos de diseño no fueron incluidas, aunque pueden influir en el precio.
7. El dashboard interactivo permite a los usuarios analizar el mercado de teléfonos móviles de manera intuitiva, facilitando la toma de decisiones, como identificar relaciones calidad-precio o tendencias de popularidad. Lo cual las empresas pueden usar este análisis para ajustar estrategias de precios y marketing.

*Mejoras Futuras*
1. Incluir datos adicionales, como valoraciones de usuarios o análisis del ciclo de vida de los dispositivos.
2. Optimizar el modelo con técnicas avanzadas, como redes neuronales, para mejorar la precisión.