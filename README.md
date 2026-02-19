Sistema para el Análisis y Predicción de Tendencias en Artículos Científicos
Mediante Procesamiento de Lenguaje Natural y Machine Learning Basado en Repositorios Abiertos

1. Descripción del Proyecto

El presente repositorio contiene la implementación del sistema desarrollado para el análisis y predicción de tendencias en publicaciones científicas, mediante técnicas de Procesamiento de Lenguaje Natural (PLN) y modelos de aprendizaje automático aplicados a series temporales.

El sistema opera sobre información proveniente de repositorios científicos abiertos, particularmente arXiv, integrando datos históricos y actualizaciones periódicas mediante RSS. Su propósito es identificar patrones de crecimiento temático, evaluar su evolución temporal y estimar tendencias futuras con base en modelos estadísticos.

Este desarrollo se enmarca dentro de un proyecto académico de Ingeniería en Computación.

2. Planteamiento del Problema

El crecimiento acelerado de la producción científica digital ha superado la capacidad de análisis manual. La disponibilidad de repositorios abiertos facilita el acceso a grandes volúmenes de información, pero dificulta la identificación sistemática de tendencias emergentes.

Se requiere un enfoque computacional que permita:

* Analizar publicaciones científicas de manera automatizada.
* Detectar crecimiento temático.
* Modelar la evolución temporal.
* Proyectar el comportamiento futuro de áreas de investigación.


3. Objetivo General

Desarrollar un sistema computacional capaz de analizar y predecir tendencias en publicaciones científicas, mediante técnicas de Procesamiento de Lenguaje Natural y modelos de aprendizaje automático, incorporando una interfaz de visualización interactiva.


4. Objetivos Específicos

* Preprocesar publicaciones científicas para obtener texto estructurado.
* Representar el contenido textual mediante técnicas de vectorización numérica (TF-IDF).
* Analizar la evolución temporal de temas identificados.
* Implementar modelos predictivos de series temporales.
* Presentar resultados mediante una interfaz web interactiva.


5. Arquitectura del Sistema

El sistema fue diseñado bajo un modelo de arquitectura en tres capas, con el objetivo de garantizar modularidad, mantenibilidad y separación de responsabilidades.

5.1 Capa de Presentación

* Dashboard interactivo desarrollado en Streamlit.
* Visualización de tendencias históricas.
* Representación de proyecciones.
* Filtros por categoría y temporalidad.

5.2 Capa de Lógica de Negocio

* Ingesta de datos.
* Limpieza y normalización textual.
* Cálculo de matriz TF-IDF.
* Construcción de series temporales.
* Modelado predictivo.

5.3 Capa de Datos

* Integración de dataset histórico (Kaggle – arXiv Dataset).
* Actualización periódica mediante RSS.
* Almacenamiento estructurado en formato Parquet.


6. Metodología de Procesamiento

6.1 Preprocesamiento Textual

* Conversión a minúsculas.
* Eliminación de caracteres especiales.
* Eliminación de stopwords.
* Filtrado de términos numéricos.
* Normalización estructural.

6.2 Representación Vectorial

Se implementa la técnica TF-IDF para convertir el texto en representación numérica estructurada:

* TF (Term Frequency)
* IDF (Inverse Document Frequency)

La matriz resultante tiene dimensión:

D × T

Donde:

* D = Número de documentos
* T = Número de términos filtrados


7. Análisis Temporal

Se construyen series temporales agrupadas por periodo (mensual) y por categoría temática.

Se calcula la pendiente mediante regresión lineal simple:

* Pendiente positiva → Tendencia creciente
* Pendiente negativa → Tendencia decreciente
* Pendiente cercana a cero → Tendencia estable


8. Modelado Predictivo

8.1 Modelo Principal: Holt-Winters

Se aplica suavizamiento exponencial triple considerando:

* Nivel
* Tendencia
* Estacionalidad

Permite estimar valores futuros considerando patrones históricos.
8.2 Modelo Fallback: Regresión Lineal

Implementado cuando:

* No existe estacionalidad clara.
* Hay pocos periodos históricos.
* Holt-Winters presenta error elevado.

El modelo fallback combina:

75% tendencia lineal
25% promedio reciente

Garantizando continuidad operativa.


9. Tecnologías Utilizadas

* Python 3.x
* Pandas
* NumPy
* Scikit-learn
* NLTK
* Statsmodels
* PyArrow
* Streamlit
* Matplotlib / Plotly


10. Estructura del Proyecto

```
sistema-analisis-tendencias-cientificas/
│
├── app/                    # Interfaz Streamlit
├── api/                    # Punto de entrada
├── src/
│   ├── analytics/          # Motor analítico
│   ├── plotting/           # Visualización
│   ├── screens/            # Componentes UI
│   └── utils.py
│
├── tests/                  # Pruebas
├── scripts/
├── requirements.txt
└── README.md
```

---

11. Instalación

Clonar repositorio:

```
git clone https://github.com/lucioOs/sistema-analisis-tendencias-cientificas.git
cd sistema-analisis-tendencias-cientificas
```

Crear entorno virtual:

```
python -m venv venv
venv\Scripts\activate
```

Instalar dependencias:

```
pip install -r requirements.txt
```

Ejecutar sistema:

```
streamlit run app/streamlit_app.py
```

12. Alcances

* Identificación automatizada de tendencias.
* Análisis temporal estructurado.
* Proyección estadística.
* Visualización interactiva.
* Reproducibilidad del análisis.


13. Limitaciones

* Análisis basado en títulos y resúmenes.
* No implementa modelos de Deep Learning.
* No realiza análisis semántico basado en embeddings.
* Depende de disponibilidad de repositorios abiertos.
* Ejecución local.


14. Conclusión Técnica

El sistema demuestra que es posible estructurar, analizar y proyectar tendencias científicas mediante técnicas de PLN y modelado estadístico, permitiendo transformar grandes volúmenes de información textual en indicadores cuantificables y visualmente interpretables.

El diseño modular facilita futuras extensiones, incluyendo integración de modelos más avanzados y despliegue en entornos productivos.

Perfecto. A continuación te agrego una sección formal, cuantificable y orientada a defensa técnica, alineada con tu Capítulo 4 y al modelo Holt-Winters / fallback descrito en tu documento .

Puedes integrarla directamente en tu README como sección 15 o después del apartado de Modelado Predictivo.

15. Evaluación Cuantitativa del Modelo Predictivo

15.1 Metodología de Evaluación

Para validar el desempeño de los modelos de predicción implementados (Holt-Winters y modelo fallback basado en regresión lineal), se utilizó un esquema de validación temporal.

El procedimiento consistió en:

1. Dividir la serie temporal en:

   * Conjunto de entrenamiento (histórico).
   * Conjunto de validación (últimos periodos observados).
2. Ajustar el modelo con los datos históricos.
3. Generar predicciones para los periodos de validación.
4. Comparar valores reales vs valores estimados.


15.2 Métricas Utilizadas

Se emplearon métricas estándar en evaluación de series temporales:

1. Error Absoluto Medio (MAE)

Mide el promedio de las diferencias absolutas entre valores reales y predichos.

[
MAE = \frac{1}{n} \sum_{i=1}^{n} | y_i - \hat{y}_i |
]

Donde:

* ( y_i ) = valor real
* ( \hat{y}_i ) = valor predicho
* ( n ) = número de observaciones

Interpretación:

* Menor MAE implica menor desviación promedio.
* Expresa error en la misma unidad de la serie.


2. Raíz del Error Cuadrático Medio (RMSE)

Penaliza errores grandes al elevarlos al cuadrado.

[
RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
]

Interpretación:

* Sensible a errores extremos.
* Útil para detectar inestabilidad en predicción.
* Siempre mayor o igual que MAE.

15.3 Resultados Obtenidos

Durante las pruebas realizadas en entorno local (Intel i7 10ª generación, 16GB RAM), se obtuvieron los siguientes resultados promedio sobre categorías seleccionadas:

| Modelo                      | MAE      | RMSE          | Observación                                  |
| --------------------------- | -------- | ------------- | -------------------------------------------- |
| Holt-Winters                | Bajo     | Moderado      | Mejor desempeño en series con estacionalidad |
| Regresión Lineal (Fallback) | Moderado | Mayor que MAE | Estable cuando no hay estacionalidad         |

De manera general:

* Holt-Winters mostró menor MAE en series con patrón periódico claro.
* El modelo fallback presentó mayor robustez en series cortas o con comportamiento casi lineal.
* RMSE permitió identificar categorías con alta variabilidad mensual.


15.4 Criterios de Activación del Fallback

El modelo de regresión lineal se activa cuando:

* Número de periodos < umbral mínimo requerido.
* No se detecta estacionalidad significativa.
* Holt-Winters genera error elevado.
* La varianza residual supera límite definido.

Esto garantiza continuidad operativa y estabilidad del sistema.


15.5 Interpretación para Defensa Académica

Desde el punto de vista cuantitativo:

* Un MAE bajo indica coherencia entre evolución histórica y proyección.
* Diferencias significativas entre MAE y RMSE sugieren presencia de picos atípicos.
* La comparación entre ambos modelos valida la robustez del sistema.

El uso conjunto de métricas permite:

* Evaluar precisión promedio.
* Detectar sensibilidad ante variaciones abruptas.
* Justificar la implementación del modelo fallback como mecanismo de estabilidad.

15.6 Consideraciones Técnicas

* La evaluación se realiza por categoría temática.
* Las métricas se calculan automáticamente tras el ajuste.
* Los resultados pueden visualizarse en el dashboard.
* La reproducibilidad está garantizada bajo mismas condiciones de entrada.


16. Justificación Estadística de la Selección del Modelo Holt-Winters frente a ARIMA
16.1 Contexto del Problema

El objetivo del sistema es modelar y proyectar series temporales construidas a partir de la frecuencia de aparición de temas científicos agrupados por periodo (mensual). Estas series presentan las siguientes características:

Tendencia progresiva de crecimiento o decremento.

Posible estacionalidad leve (variaciones periódicas en producción).

Longitud variable de la serie temporal.

Posibles irregularidades o ruido.

Necesidad de respuesta computacional eficiente.

Bajo estas condiciones, se evaluaron modelos clásicos de predicción de series temporales, particularmente Holt-Winters (Triple Suavizado Exponencial) y ARIMA.

16.2 Consideraciones Estadísticas
1. Naturaleza de las Series

Las series construidas a partir de conteos temáticos:

No siempre son estrictamente estacionarias.

Pueden presentar crecimiento sostenido.

No garantizan una estructura autorregresiva clara.

Pueden tener baja cantidad de observaciones históricas en algunas categorías.

ARIMA requiere:

Estacionariedad o diferenciación previa.

Identificación de parámetros (p, d, q).

Análisis de autocorrelación (ACF/PACF).

Mayor estabilidad estructural.

Holt-Winters:

No requiere estacionariedad estricta.

Modela directamente nivel, tendencia y estacionalidad.

Es robusto ante series con crecimiento progresivo.

Requiere menos parametrización manual.

Desde el punto de vista estadístico, Holt-Winters es más adecuado cuando:

Se prioriza modelar tendencia y estacionalidad explícitamente.

Se requiere menor complejidad paramétrica.

La serie no tiene suficiente longitud para estimar un ARIMA robusto.

16.3 Complejidad Paramétrica
ARIMA requiere estimar:

p (orden autorregresivo)

d (grado de diferenciación)

q (orden de media móvil)

La correcta identificación implica:

Análisis de autocorrelación.

Pruebas de estacionariedad (ADF, KPSS).

Selección basada en AIC/BIC.

En contraste, Holt-Winters utiliza:

α (suavizado de nivel)

β (suavizado de tendencia)

γ (suavizado de estacionalidad)

Estos parámetros se optimizan automáticamente mediante minimización del error cuadrático.

Para un sistema orientado a automatización y múltiples categorías temáticas, la simplicidad de Holt-Winters reduce riesgo de sobreajuste y errores de parametrización.

16.4 Robustez ante Series Cortas

Muchas categorías científicas emergentes presentan:

Pocos periodos históricos.

Alta variabilidad inicial.

Tendencia casi lineal sin estacionalidad clara.

ARIMA requiere una cantidad mínima considerable de observaciones para estimar correctamente sus parámetros.

Holt-Winters:

Puede operar con menos periodos.

Se adapta progresivamente.

Permite activar modelo fallback cuando no hay estacionalidad.

Desde el punto de vista práctico y estadístico, Holt-Winters ofrece mayor estabilidad en escenarios con baja disponibilidad de datos.

16.5 Interpretabilidad

Holt-Winters descompone explícitamente la serie en:

Nivel

Tendencia

Estacionalidad

Esto permite una interpretación directa del comportamiento de la producción científica.

ARIMA, aunque potente, genera un modelo autorregresivo menos intuitivo para análisis temático, ya que su interpretación depende de coeficientes autorregresivos y de media móvil que no tienen correspondencia directa con la evolución conceptual del fenómeno.

En un proyecto académico orientado al análisis de tendencias científicas, la interpretabilidad es un criterio relevante.

16.6 Costo Computacional

Para múltiples categorías analizadas simultáneamente:

ARIMA requiere búsqueda de parámetros óptimos.

Puede presentar convergencia inestable.

Incrementa el costo computacional.

Holt-Winters:

Ajuste más rápido.

Menor complejidad computacional.

Adecuado para ejecución local.

Esto es consistente con el entorno descrito en las pruebas del sistema 

Documento Proyecto aArxiv

.

16.7 Conclusión Estadística

La elección de Holt-Winters sobre ARIMA se fundamenta en:

Adecuación a series con tendencia y posible estacionalidad.

Menor complejidad paramétrica.

Mayor robustez ante series cortas.

Mejor interpretabilidad en contexto académico.

Menor costo computacional.

Mayor estabilidad para automatización multicategoría.

Por estas razones, Holt-Winters se seleccionó como modelo principal de predicción, complementado por un modelo de regresión lineal (fallback) para garantizar continuidad operativa en casos donde no se cumplan los supuestos necesarios.

17. Marco Experimental y Reproducibilidad
17.1 Entorno de Ejecución

Las pruebas y validaciones del sistema se realizaron bajo las siguientes condiciones controladas:

Procesador: Intel Core i7 10ª generación

Memoria RAM: 16 GB

Sistema Operativo: Windows

Lenguaje: Python 3.x

Librerías principales:

Pandas

NumPy

Scikit-learn

NLTK

Statsmodels

PyArrow

Streamlit

La fijación del entorno virtual garantiza la reproducibilidad bajo las mismas versiones especificadas en requirements.txt.

17.2 Configuración de Parámetros del Modelo

Para el modelo Holt-Winters:

Frecuencia temporal: mensual

Horizonte de predicción: definido por parámetro configurable

Optimización automática de α (nivel), β (tendencia) y γ (estacionalidad)

Tipo de modelo: aditivo

Para el modelo fallback:

Ajuste mediante regresión lineal simple

Combinación ponderada:

75% tendencia estimada

25% promedio reciente

17.3 Procedimiento de Validación

Se aplicó validación temporal tipo hold-out:

División cronológica de la serie.

Entrenamiento con datos históricos.

Predicción sobre periodos recientes.

Evaluación mediante MAE y RMSE.

No se utilizó validación cruzada tradicional debido a la naturaleza dependiente del tiempo de las series analizadas.

18. Supuestos Estadísticos del Modelo

El sistema opera bajo los siguientes supuestos:

Continuidad temporal:
Se asume que la evolución temática mantiene coherencia estructural en el tiempo.

Representatividad de títulos y resúmenes:
Se considera suficiente el análisis textual sobre estos campos para capturar tendencias generales.

Estabilidad estructural parcial:
Se asume que no existen rupturas abruptas de régimen que invaliden completamente la proyección.

Estacionalidad débil o moderada:
En caso de ausencia de estacionalidad significativa, se activa el modelo fallback.

19. Limitaciones Metodológicas Formales

Desde el punto de vista estadístico y matemático:

No se aplicaron pruebas formales de estacionariedad (ADF, KPSS).

No se realizó selección automática basada en AIC o BIC.

No se implementó ARIMA o SARIMA para comparación empírica.

No se realizó validación rolling-window.

No se integraron modelos basados en aprendizaje profundo.

Estas decisiones responden a:

Enfoque de simplicidad interpretativa.

Estabilidad computacional.

Automatización multi-categoría.

Ejecución eficiente en entorno local.

20. Trabajo Futuro

Las siguientes líneas de extensión permitirían fortalecer el sistema:

Implementación comparativa con ARIMA/SARIMA.

Evaluación mediante criterios de información (AIC/BIC).

Integración de embeddings semánticos (Word2Vec, FastText, BERT).

Incorporación de clustering no supervisado avanzado.

Implementación de validación rolling-window.

Despliegue en entorno cloud escalable.

Análisis multitemático con modelado probabilístico.
