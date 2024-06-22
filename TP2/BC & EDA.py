# Databricks notebook source
# MAGIC %md
# MAGIC <div style="text-align: center;">
# MAGIC     <h1> Trabajo Práctico 2 </font></h1>
# MAGIC     <h2>Caso de Negocio y Análisis Exploratorio </font></h2>
# MAGIC     <h3>Abril Noguera - Abril Schafer - Ian Dalton</font></h3>
# MAGIC </div>

# COMMAND ----------

# Importación de Librerias
from pyspark import SparkFiles
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, StringIndexer, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.stat import ChiSquareTest
from pyspark.ml.evaluation import RegressionEvaluator
import os
import subprocess
import uuid
from functools import reduce
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_ind, f_oneway

# COMMAND ----------

# Configurar los gráficos de Seaborn
sns.set(style="whitegrid")
color = '#FF385C'
color2 = '#ff5a5f'
color3 = '#f6a8ac'
color4 = '#6a6a6a'

# COMMAND ----------

# MAGIC %md
# MAGIC ## Caso de Negocio: Optimización de la Ocupación y Estrategias de Precios en Airbnb
# MAGIC
# MAGIC ### Problema de Negocio
# MAGIC Los anfitriones de Airbnb en diversas metrópolis globales como Buenos Aires, Londres, París, Roma y Nueva York buscan continuamente optimizar la ocupación de sus propiedades para maximizar ingresos y mejorar la satisfacción del cliente. El desafío consiste en adaptar las estrategias de precios y mejorar las características de las propiedades de manera que respondan a las dinámicas del mercado y las preferencias cambiantes de los huéspedes en diferentes culturas y condiciones económicas.
# MAGIC
# MAGIC ### Objetivo
# MAGIC Desarrollar un modelo predictivo de machine learning utilizando Spark que permita a los anfitriones de Airbnb en estas ciudades predecir la ocupación de sus propiedades basándose en datos históricos y tendencias de mercado. El objetivo final es utilizar estos insights para recomendar estrategias de precios que maximicen la ocupación y, consecuentemente, los ingresos.
# MAGIC
# MAGIC ### Datos Disponibles
# MAGIC El dataset contiene información detallada sobre las propiedades listadas en estas cinco ciudades, totalizando 280,327 registros. Los datos incluyen ubicación, detalles del anfitrión, tipo de propiedad, capacidad, precios, mínimos y máximos de noches, disponibilidad, número de reseñas y puntuaciones en diferentes aspectos, entre otros.
# MAGIC
# MAGIC Los datos representan una oportunidad para analizar tendencias de mercado, patrones de ocupación, preferencias de los clientes y otros factores que influyen en la ocupación y rentabilidad de las propiedades.
# MAGIC
# MAGIC ### Variable Target
# MAGIC La variable target para el modelo predictivo será la **ocupación** (derivada de las columnas de disponibilidad), que se buscará predecir basándose en características del listado y otros indicadores de desempeño histórico.
# MAGIC
# MAGIC ### Población Objetivo
# MAGIC La población objetivo incluye todos los listados de propiedades en Airbnb de estas ciudades, recopilados a través de un scrapping realizado en marzo de 2024. Este dataset integral representa todas las propiedades disponibles para alquiler en ese momento, proporcionando una base exhaustiva sobre la cual entrenar el modelo.
# MAGIC
# MAGIC ### Periodos Considerados en el Dataset
# MAGIC El dataset incluye datos recopilados en un scrapping específico realizado en marzo de 2024, capturando así todas las propiedades activas en Airbnb en ese período.
# MAGIC
# MAGIC ### Estrategias y Soluciones Propuestas
# MAGIC Desarrollar un modelo de machine learning que pueda predecir la ocupación de las propiedades basándose en factores como la ubicación, tipo de propiedad, temporada del año y calificaciones de reseñas. Basándose en las predicciones de ocupación, el modelo también sugerirá ajustes de precios y otras estrategias de optimización para ayudar a los anfitriones a tomar decisiones basadas en datos que maximicen la ocupación y rentabilidad en un contexto global diverso.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Análisis y Limpieza de Datos
# MAGIC ### Carga de Datos
# MAGIC Cargar el dataset utilizando Spark, configurando la lectura para manejar encabezados y tipos de datos automáticamente.

# COMMAND ----------

def load_and_get_df(url: str, alias: str) -> DataFrame:
    '''
    Función para cargar un archivo desde un URL, especificando un alias 
    - Input: url de descarga y nombre de la ciudad como alias.
    - Output: df
    '''
    # Generar un nombre de archivo único para evitar colisiones en el espacio de nombres
    unique_filename = "listings_" + alias.replace(' ', '_').lower() + "_" + str(uuid.uuid4()) + ".csv.gz"
    
    # Añadir el archivo al contexto de Spark con un identificador único
    local_path = "/tmp/" + unique_filename  # Camino local temporal para el archivo
    # Descargar el archivo utilizando wget o curl (necesitas tener acceso a bash)
    os.system(f"wget -O {local_path} {url}")
    
    # Añadir el archivo al SparkContext
    spark.sparkContext.addFile(local_path)
    
    # Leer el archivo como un DataFrame de Spark
    df = spark.read \
        .option("sep", ",") \
        .option("quote", "\"") \
        .option("escape", "\"") \
        .option("multiLine", "true") \
        .option("encoding", "UTF-8") \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .csv("file://" + SparkFiles.get(unique_filename))
    
    # Añadir una columna que indique la ciudad
    df = df.withColumn("city", lit(alias))
    print(f"Registros en {alias}: {df.count()}")
    return df

# URLs de los datasets
urls = {
    "Buenos Aires": "https://data.insideairbnb.com/argentina/ciudad-aut%C3%B3noma-de-buenos-aires/buenos-aires/2024-04-28/data/listings.csv.gz",
    "London": "https://data.insideairbnb.com/united-kingdom/england/london/2024-03-19/data/listings.csv.gz",
    "Paris": "https://data.insideairbnb.com/france/ile-de-france/paris/2024-03-16/data/listings.csv.gz",
    "Rome": "https://data.insideairbnb.com/italy/lazio/rome/2024-03-22/data/listings.csv.gz",
    "New York": "https://data.insideairbnb.com/united-states/ny/new-york-city/2024-05-03/data/listings.csv.gz"
}

# Cargar y unir todos los dataframes
dfs = [load_and_get_df(url, city) for city, url in urls.items()]
df = reduce(DataFrame.unionByName, dfs)
print(f"Total de registros combinados: {df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualización de la Estructura de Datos
# MAGIC Revisar el esquema del DataFrame para comprender la estructura y los tipos de datos disponibles.

# COMMAND ----------

# Muestra de la base
df.display()

# COMMAND ----------

# Convertir host_response_rate y host_acceptance_rate de porcentaje a número
df = df.withColumn("host_response_rate", regexp_replace(col("host_response_rate"), "%", "").cast("double") / 100)\
    .withColumn("host_acceptance_rate", regexp_replace(col("host_acceptance_rate"), "%", "").cast("double") / 100)\
    .withColumn("price", regexp_replace(col("price"), "[\$,]", "").cast("double"))


# COMMAND ----------

# Muestra de la estructura de la base
df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Análisis Preliminar de Datos
# MAGIC - **Visualización del Total de Registros y Columnas**: Evaluar la cantidad de datos disponibles.
# MAGIC - **Análisis Descriptivo de Datos Numéricos**: Descripción de valores numéricos

# COMMAND ----------

# Se obtienen las columnas categoricas
categoric_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, StringType)]
print("Las columnas de tipo categoricas son: ", categoric_cols, '\n')

# Se obtienen las columnas númericas
numeric_cols = [f.name for f in df.schema.fields if (isinstance(f.dataType, DoubleType)) | (isinstance(f.dataType, IntegerType))]
print("Las columnas de tipo numéricas son: ", numeric_cols)

# COMMAND ----------

# Descripción de variables númericas
df.select(numeric_cols).describe().display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Análisis de Valores Faltantes
# MAGIC - **Identificar Columnas con Valores Faltantes**: Determinar qué columnas contienen valores faltantes y en qué proporción.
# MAGIC - **Evaluar el Impacto**: Considerar cómo la falta de datos puede impactar los análisis posteriores y las decisiones basadas en datos.
# MAGIC - **Estrategias de Manejo**: Definir cómo se manejarán los valores faltantes, ya sea mediante imputación, eliminación o  otro método adecuado.

# COMMAND ----------

# Hay columnas de texto que tiene un unico simbolo que serán representadas como nulos.
for column_name in categoric_cols:
        # Verificar si la columna es de tipo String
        if isinstance(df.schema[column_name].dataType, StringType):
            # Reemplazar '.' o '-' con nulos si son los únicos caracteres en el campo
            df = df.withColumn(column_name, when(col(column_name).rlike("^[.-]$"), None).otherwise(col(column_name)))

# Hay columnas con valor "N/A" que seran interpretadas como nulos. 
df = df.na.replace("N/A", None)

# COMMAND ----------

# Muestra de nulos por columna
df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).show(vertical=True)

# COMMAND ----------

# Se mantienen unicamente los registros que tengan precio, dado que es la variable a predecir
df = df.filter(col("price").isNotNull())

# COMMAND ----------

# Muestra de nulos por columna
df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).show(vertical=True)

# COMMAND ----------

# Se eliminan las columnas donde el 90% de esus valores sean nulos. 
df = df.drop("neighbourhood_group_cleansed", "license", 'calendar_updated')

# Se mantienen como vacios las descripciones, dado que puede ser un indicador importante en la performance del alquiler.
df = df.fillna('', ["description", "neighborhood_overview", 'host_about'])

# La columna neighbourhood no es muy representativa al tratarse de información de CABA, por lo que se mantiene unicamente neighbourhood_cleansed.
df = df.drop("neighbourhood")

# Para la columna bathrooms se intenta obtener el valor de bathrooms_text, si quedan más nulos se utiliza la mediana. 
df = df.withColumn("extracted_bathrooms", regexp_extract(col("bathrooms_text"), r"(\d+)", 1).cast("float"))\
            .withColumn("bathrooms", when(col("bathrooms").isNull(), col("extracted_bathrooms")).otherwise(col('bathrooms')))\
            .drop("extracted_bathrooms")

# Los valores que refieren a las reviews y estan nulos son por no tener ninguna review aún, por lo que se imputa con la media. 
df = df.withColumn("first_review", coalesce("first_review", "host_since"))\
            .withColumn("last_review", coalesce("last_review", "host_since"))

# Se imputan los valores con la mediana
imputerMediana = Imputer(
    inputCols= ['bathrooms', 'bedrooms', 'beds'], 
    strategy='median',
    outputCols=[ 'bathrooms', 'bedrooms', 'beds']
)
df = imputerMediana.fit(df).transform(df)

# Se imputan los valores con la media
imputerMedia = Imputer(
    inputCols= ['host_response_rate', 'host_acceptance_rate', 'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value', 'reviews_per_month'], 
    strategy='mean',
    outputCols=['host_response_rate', 'host_acceptance_rate', 'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value', 'reviews_per_month']
)
df = imputerMedia.fit(df).transform(df)

# Se imputan los valores con la moda
def ImputerModa(df, columns):
    '''
    Función para imputer valores con la moda (No definida en Spark)
    - Input: df y columnas a imputar.
    - Output: df
    '''
    for c in columns:
        moda = df.filter(col(c).isNotNull()).groupBy(c).agg(count("*").alias("cantidad")).orderBy(col("cantidad").desc()).limit(1).select(c).first()[0]
        df = df.fillna(moda, c)
    return df

df = ImputerModa(df, ["host_location", "host_neighbourhood", "host_response_time", "host_is_superhost", "bathrooms_text"])

# Se eliminan los registros con name y picture_url nulos
df = df.filter(col("name").isNotNull()).filter(col("picture_url").isNotNull())

# Se determina has_availiability condicional segun las otras columnas de availabilty
df = df.withColumn("has_availability", when((col('availability_30') == 0) & (col('availability_60') == 0) & (col('availability_90') == 0) & (col('availability_365') == 0)  & (col('has_availability').isNull()), "f").otherwise(col('has_availability'))).fillna("t", "has_availability")



# COMMAND ----------

# Muestra de nulos por columna
df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).show(vertical=True)

# COMMAND ----------

print(f"El total de registros luego de la limpieza de nulos es: {df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ###  Limpieza de Datos de Texto
# MAGIC - **Eliminar Etiquetas HTML:** Antes de proceder con el análisis exploratorio más detallado, se deben limpiar las variables de texto que pueden contener etiquetas de HTML, lo cual puede distorsionar las interpretaciones de los datos.
# MAGIC - **Convertir Columnas a Dummies:** Procesar los datos de `host_verifications` para convertir esta columna de listas de verificaciones en múltiples columnas binarias (dummies), cada una representando una verificación posible.
# MAGIC

# COMMAND ----------

def remove_html_tags(df):
    '''
    Función para eliminar etiquetas de HTML de las variables categoricas. 
    - Input: df
    - Output: df
    '''
    for column_name in df.columns:
        # Chequear si la columna es de tipo string
        if isinstance(df.schema[column_name].dataType, StringType):
            # Limpiar las etiquetas HTML de las columnas de tipo string
            df = df.withColumn(column_name, regexp_replace(col(column_name), "<.*?>", " "))
            # Limpiar todos los caracteres no ASCII
            df = df.withColumn(column_name, regexp_replace(col(column_name), "[^\x00-\x7F]+", " "))
            # Normalizar espacios: convertir múltiples espacios a un solo espacio
            df = df.withColumn(column_name, regexp_replace(col(column_name), "\\s+", " "))
    return df

# COMMAND ----------

# Eliminación de etiquetas de HTML
df = remove_html_tags(df)

# COMMAND ----------

# Limpiar la columna y convertirla en un array eliminando comillas y corchetes
df = df.withColumn("host_verifications", regexp_replace(col("host_verifications"), "[\\[\\]'\" ]", ""))\
        .withColumn("host_verifications", split(col("host_verifications"), ","))

# Extraer los tipos únicos de la columna
unique_types = df.select(explode(col("host_verifications")).alias("item"))\
                    .filter(col("item") != "")\
                    .distinct()\
                    .rdd.flatMap(lambda x: x)\
                    .collect()
print(f"Los valores unicos de host_verifications son: {unique_types}")

# Crear columnas dummies para cada tipo único
for item in unique_types:
    if item: 
        df = df.withColumn("verif_" + item.replace(" ", "_"), array_contains(col("host_verifications"), item).cast("int"))

df = df.drop("host_verifications")

# COMMAND ----------

# Preprocesar Amenities
df = df.withColumn('amenities', regexp_replace(col('amenities'), '[^a-zA-Z ,]', ''))\
    .withColumn('amenities', regexp_replace(col('amenities'), "\\s+", " "))\
    .withColumn('amenities', split(trim(col('amenities')), ", "))

# COMMAND ----------

# MAGIC %md
# MAGIC La variable `price` en el dataset de Airbnb está expresada en la moneda local de cada ciudad. Esto presenta un desafío para el análisis agregado y comparativo entre ciudades que están en diferentes regiones geográficas y económicas.
# MAGIC
# MAGIC Se convertiran todos los precios a una moneda común, específicamente el dólar estadounidense (USD), utilizando los tipos de cambio vigentes en las fechas de scrapping. Esta estandarización permitirá realizar comparaciones directas y más precisas entre las propiedades en diferentes ciudades.
# MAGIC

# COMMAND ----------

# Tasas de cambio aproximadas en la fecha de scrapping
tasas_cambio = {
    "Buenos Aires": 857.29,  # ARS a USD 2024-04-28
    "London": 0.79,       # GBP a USD 2024-03-19
    "Paris": 0.92,         # EUR a USD 2024-03-16
    "Rome": 0.92,          # EUR a USD 2024-03-22
    "New York": 1         # USD a USD (base)
}

def convert_currency(price, city):
    '''
    Función para convertir el precio a dolares.
    - Input: valor de precio en moneda local y ciudad
    - Output: valor de precio en dolares
    '''
    rate = tasas_cambio.get(city, 1)  # Uso 1 como tasa de cambio predeterminada si la ciudad no está en el diccionario
    return price / rate

# UDF para convertir la moneda
udf_convert_currency = udf(convert_currency, DoubleType())

# Aplicar la conversión de moneda
df = df.withColumn('price', udf_convert_currency(col('price'), col('city')))

# COMMAND ----------

# MAGIC %md
# MAGIC Convertimos las variables categóricas `host_is_superhost`, `host_has_profile_pic`, `host_identity_verified` y `instant_bookable` de 't' (True) y 'f' (False) a valores booleanos. 

# COMMAND ----------

# Conversión de 't'/'f' a 1/0
df = df.withColumn('host_is_superhost', when(col('host_is_superhost') == 't', 1).otherwise(0))\
        .withColumn('host_has_profile_pic', when(col('host_has_profile_pic') == 't', 1).otherwise(0))\
        .withColumn('host_identity_verified', when(col('host_identity_verified') == 't', 1).otherwise(0))\
        .withColumn('instant_bookable', when(col('instant_bookable') == 't', 1).otherwise(0))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Definición de Variable Target para el Modelo Predictivo
# MAGIC
# MAGIC El objetivo del modelo predictivo será estimar la tasa de ocupación de las propiedades, la cual se definirá como el complemento del porcentaje de días disponibles en un período futuro determinado. Esta métrica permite modelar directamente la demanda o popularidad de una propiedad y es crucial para la toma de decisiones operativas y estratégicas.
# MAGIC
# MAGIC Para calcular la tasa de ocupación, utilizaremos las siguientes columnas de disponibilidad:
# MAGIC - `has_availability`: Indica si la propiedad está disponible para ser reservada (t para sí, f para no).
# MAGIC - `availability_30`: Disponibilidad de la propiedad en los próximos 30 días.
# MAGIC - `availability_60`: Disponibilidad de la propiedad en los próximos 60 días.
# MAGIC - `availability_90`: Disponibilidad de la propiedad en los próximos 90 días.
# MAGIC - `availability_365`: Disponibilidad de la propiedad en los próximos 365 días.
# MAGIC
# MAGIC El modelo se enfocará en un horizonte de predicción a mediano plazo (60 días). Esta decisión se basa en:
# MAGIC - **Equilibrio entre precisión y utilidad operativa:** Un horizonte de 60 días proporciona una predicción suficientemente anticipada para permitir ajustes operativos, sin sacrificar significativamente la precisión debido a la incertidumbre de pronosticar a más largo plazo.
# MAGIC - **Optimización de estrategias de precios y promociones:** El período seleccionado permite implementar o ajustar estrategias de precios y promociones con suficiente anticipación, lo cual es esencial para optimizar la ocupación y los ingresos.
# MAGIC - **Adaptación a patrones estacionales:** Permite adaptarse a cambios estacionales que pueden afectar significativamente la demanda de alojamiento.

# COMMAND ----------

# Definición de variable target: Calcular la tasa de ocupación basada en la disponibilidad a 60 días
df = df.withColumn("occupation_rate", expr("1 - (availability_60 / 60)"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Análisis Exploratorio de Datos
# MAGIC ### Objetivos del Análisis
# MAGIC - **Identificar patrones**: Comprender los patrones de ocupación, precios, y preferencias de los usuarios.
# MAGIC - **Segmentación de propiedades**: Clasificar las propiedades en diferentes grupos basados en características como ubicación, tipo, y capacidad.
# MAGIC - **Optimización de recursos**: Identificar áreas de mejora en la gestión de las propiedades para optimizar la rentabilidad.

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1. Comprender el Dataset
# MAGIC
# MAGIC **Preguntas a responder:**
# MAGIC - ¿Cuáles son las principales características de las propiedades listadas en Airbnb en las cinco ciudades seleccionadas?
# MAGIC - ¿Cómo están distribuidas las propiedades en términos de ubicación, tipo de propiedad, capacidad y precio?
# MAGIC
# MAGIC **Análisis:**
# MAGIC - Resumen estadístico de las variables principales.
# MAGIC - Distribución de propiedades por ciudad.
# MAGIC - Tipos de propiedades más comunes en cada ciudad.
# MAGIC - Distribución de capacidad y precios por ciudad.

# COMMAND ----------

# Agrupar por ciudad y contar el número de propiedades
city_counts = df.groupBy('city').count().toPandas()

# Crear gráfico de barras
plt.figure(figsize=(19, 6))
plt.bar(city_counts['city'], city_counts['count'], color=color)
plt.xlabel('Ciudad')
plt.ylabel('Número de Listados')
plt.title('Número de Listados por Ciudad')
plt.xticks(rotation=0)
plt.grid(False)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Interpretación:**
# MAGIC París y Londres tienen el mayor número de listados, llegando cada una los 60,000 listados. Buenos Aires sigue con aproximadamente un poco más de la mitad del número de listados que París y Londres. Roma y Nueva York tienen un número similar de listados, con aproximadamente 30,000 cada una. Esto sugiere que París y Londres son los mercados más grandes en términos de número de propiedades listadas en Airbnb, lo que puede indicar una mayor demanda o una mayor oferta de alojamiento en estas ciudades.

# COMMAND ----------

# Agrupar por tipo de habitación y contar el número de propiedades
room_type_counts = df.groupBy('room_type').count().toPandas()

# Crear gráfico de barras
plt.figure(figsize=(19, 6))
plt.bar(room_type_counts['room_type'], room_type_counts['count'], color=color)
plt.xlabel('Tipo de Habitación')
plt.ylabel('Número de Listados')
plt.title('Número de Listados por Tipo de Habitación')
plt.xticks(rotation=0)
plt.grid(False)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Interpretación:**
# MAGIC La mayoría de los listados son para "Entire home/apt" (hogares o apartamentos completos), con más de 150,000 listados. "Private room" (habitaciones privadas) es el siguiente tipo más común, con aproximadamente 50,000 listados. Los tipos de habitación "Shared room" y "Hotel room" tienen un número insignificante de listados en comparación con los otros tipos. Esto sugiere que los huéspedes de Airbnb prefieren tener la propiedad completa o una habitación privada, en lugar de compartir el espacio con otros huéspedes o alojarse en un entorno similar a un hotel.

# COMMAND ----------

# Obtener los datos para disponibilidad y precios por ciudad
availability_desc = df.groupBy('city').agg({'availability_365': 'mean'}).toPandas()
price_desc = df.groupBy('city').agg({'price': 'mean'}).toPandas()

# Combinar los datos en un solo DataFrame
combined_df = pd.merge(availability_desc, price_desc, on='city')
combined_df.columns = ['city', 'avg_availability_365', 'avg_price']

# Crear el gráfico con dos ejes Y
fig, ax1 = plt.subplots(figsize=(19, 6))

# Primer eje Y para disponibilidad
ax1.set_xlabel('Ciudad')
ax1.set_ylabel('Disponibilidad Promedio (365 días)')
ax1.bar(combined_df['city'], combined_df['avg_availability_365'], color=color3, label='Disponibilidad Promedio')
ax1.tick_params(axis='y')
ax1.set_xticklabels(combined_df['city'], rotation=0)

# Segundo eje Y para precios
ax2 = ax1.twinx()
ax2.set_ylabel('Precio Promedio')
ax2.plot(combined_df['city'], combined_df['avg_price'], color=color, marker='o', label='Precio Promedio')
ax2.tick_params(axis='y')

# Título del gráfico
fig.suptitle('Disponibilidad y Precio Promedio por Ciudad')

# Mostrar el gráfico
fig.tight_layout()
plt.grid(False)
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC **Interpretación:**
# MAGIC Buenos Aires y Nueva York tienen la mayor disponibilidad promedio con más de 200 días al año. París tiene la menor disponibilidad promedio con casi 150 días al año. En cuanto al precio promedio, Paris tiene el precio más alto, superando los 300. Nueva York, Roma y Londres tienen precios más bajos en comparación con Paris, mientras que Buenos Aires tiene el precio promedio más bajo. La alta disponibilidad en Buenos Aires podría sugerir una menor demanda en comparación con otras ciudades. En contraste, Paris tiene alta demanda (baja disponibilidad) y precios altos, indicando un mercado más competitivo.

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2. Identificación de Outliers
# MAGIC
# MAGIC Para visualizar los outliers en la variable `price`, utilizamos un boxplot. Este gráfico nos permite identificar visualmente los valores atípicos que se encuentran fuera del rango esperado.

# COMMAND ----------

# Convertir la columna de precios y ciudad a un DataFrame de Pandas
price_df = df.select('price', 'city').toPandas()

# Crear un boxplot para visualizar los outliers con Seaborn
plt.figure(figsize=(20, 8)) 
sns.boxenplot(y=price_df["city"], x=price_df["price"], palette="Set1") 
plt.title('Boxplot de Precios por Ciudad')
plt.xlabel('Precio')
plt.ylabel('Ciudad')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC El boxplot de precios por ciudad muestra que hay una cantidad significativa de outliers en la variable `price` para cada una de las ciudades analizadas. Estos outliers se extienden mucho más allá del rango intercuartílico, indicando valores de precios extremadamente altos en comparación con el rango típico de precios.
# MAGIC
# MAGIC Dado que hay una gran cantidad de outliers en la variable `price`, se procederá a hacer un corte para remover estos valores atípicos. Esto permitirá limpiar el dataset y mejorar la calidad de los análisis y predicciones posteriores. La remoción de outliers se realizará utilizando el método de IQR (Interquartile Range), definiendo los outliers como aquellos valores que están por debajo del primer cuartil (Q1) menos 1.5 veces el IQR o por encima del tercer cuartil (Q3) más 1.5 veces el IQR.
# MAGIC
# MAGIC Además, se debe asegurar que el precio minimo sea 1. Los valores de precio negativos no son correctos en este análisis. 

# COMMAND ----------

# Calcular Q1 y Q3
q1, q3 = df.select(
    percentile_approx("price", 0.25).alias("q1"),
    percentile_approx("price", 0.75).alias("q3")
).first()

# Calcular IQR
iqr = q3 - q1

# Definir los límites inferior y superior para los outliers
lower_bound = 1
upper_bound = q3 + 1.5 * iqr

print(f'Límite inferior: {lower_bound}')
print(f'Límite superior: {upper_bound}')

# COMMAND ----------

# Filtrar el DataFrame para remover los outliers
df = df.filter((col('price') >= lower_bound) & (col('price') <= upper_bound))

# Mostrar algunas estadísticas del DataFrame filtrado
df.select('price').describe().show()

# COMMAND ----------

# Gráfico de dispersión para cada ciudad
cities = df.select('city').distinct().rdd.flatMap(lambda x: x).collect()
plt.figure(figsize=(19, 10))
for i, city in enumerate(cities):
    plt.subplot(3, 2, i+1)
    city_df = df.filter(df.city == city).select('price', 'availability_365').toPandas()
    plt.scatter(city_df['price'], city_df['availability_365'], color=color, alpha=0.5)
    plt.title(f'Relación entre Precio y Disponibilidad en {city}')
    plt.xlabel('Precio')
    plt.ylabel('Disponibilidad (365 días)')
plt.tight_layout()
plt.show()

# COMMAND ----------

# Calcular el coeficiente de correlación
for city in cities:
    corr_value = df.filter(df.city == city).select(corr('price', 'availability_365')).collect()[0][0]
    print(f'Coeficiente de correlación entre Precio y Disponibilidad en {city}: {corr_value}')

# COMMAND ----------

# MAGIC %md
# MAGIC Los coeficientes de correlación calculados son todos relativamente bajos, lo que indica una relación débil entre el precio y la disponibilidad en todas las ciudades analizadas. Ninguna de las ciudades muestra una fuerte correlación positiva o negativa entre estas dos variables.
# MAGIC
# MAGIC Aunque todas las correlaciones son bajas, se observa cierta variación entre las ciudades. Rome muestra la correlación más alta (0.0989), mientras que New York presenta la correlación más baja (0.0255).

# COMMAND ----------

# MAGIC %md
# MAGIC Para visualizar los outliers en las variables `bathrooms`, `bedrooms`, `beds` y `accommodates`, utilizamos violinplots. Estos gráficos nos permiten identificar visualmente los valores atípicos que se encuentran fuera del rango esperado.

# COMMAND ----------

# Variables a analizar
variables = ['bathrooms', 'bedrooms', 'beds', 'accommodates']

plt.figure(figsize=(20, 10))
for i, var in enumerate(variables):
    plt.subplot(2, 2, i+1)
    var_df = df.select(var).toPandas()
    sns.violinplot(y=var_df[var], palette=[color])
    plt.title(f'Violin Plot de {var}')
    plt.ylabel(var)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Las visualizaciones sugieren que las variables `bathrooms`, `bedrooms`, `beds` y `accommodates` contienen varios outliers que podrían distorsionar los análisis y modelos predictivos.
# MAGIC
# MAGIC Se definio un umbral de maximo 10 para cada una de las variables, eliminando los registros que excedan este umbral.

# COMMAND ----------

# Definir los umbrales máximos
max_umbral = 10

# Filtrar el DataFrame para remover los outliers
df = df.filter((col('bathrooms') <= max_umbral) &
                (col('bedrooms') <= max_umbral) &
                (col('beds') <= max_umbral) &
                (col('accommodates') <= max_umbral))

# Mostrar algunas estadísticas del DataFrame filtrado
df.select('bathrooms', 'bedrooms', 'beds', 'accommodates').describe().show()

# COMMAND ----------

# MAGIC %md
# MAGIC Se analizaran además la presencia de outliers en `minimum_nights` y `maximum_nights` realizando boxplots.

# COMMAND ----------

# Variables a analizar
variables = ['minimum_nights', 'maximum_nights', 'minimum_minimum_nights', 'maximum_minimum_nights', 'minimum_maximum_nights', 'maximum_maximum_nights']

plt.figure(figsize=(20, 10))
for i, var in enumerate(variables):
    plt.subplot(2, 3, i+1)
    var_df = df.select(var).toPandas()
    sns.violinplot(y=var_df[var], palette=[color])
    plt.title(f'Violin Plot de {var}')
    plt.ylabel(var)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC La visualización indica la presencia de valores extremos que distorsionan la interpretación de los datos. Para tratar los valores extremos en la variable `minimum_nights`, se ha decidido imputar los valores que exceden un umbral razonable (200 noches) con este valor máximo. Esto se hace para mantener la coherencia de los datos y evitar sesgos en el análisis.
# MAGIC
# MAGIC En el caso de `maximum_nights` los valores extremos pueden traer valor al no ser común que se apliquen restricciones en las noches maximas. Por lo que se mantiene.

# COMMAND ----------

# Definir el umbral máximo para imputación
max_minimum_nights = 200

# Imputar valores extremos en minimum_nights
df = df.withColumn('minimum_nights', when(col('minimum_nights') > max_minimum_nights, max_minimum_nights).otherwise(col('minimum_nights')))\
       .withColumn('minimum_minimum_nights', when(col('minimum_minimum_nights') > max_minimum_nights, max_minimum_nights).otherwise(col('minimum_minimum_nights')))\
       .withColumn('maximum_minimum_nights', when(col('maximum_minimum_nights') > max_minimum_nights, max_minimum_nights).otherwise(col('maximum_minimum_nights')))

# Mostrar algunas estadísticas del DataFrame imputado
df.select('minimum_nights').describe().show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3. Correlación con Ocupación
# MAGIC Exploramos la relación entre las variables númericas con la variable target para identificar patrones o tendencias que puedan informar decisiones de negocio fundamentadas.

# COMMAND ----------

# Seleccionar las columnas numéricas y la variable objetivo
columns_to_remove = ['longitude', 'latitude']
numerical_columns = [col for col in numeric_cols if col not in columns_to_remove]

# Definimos la variable target
target_column = 'occupation_rate'

# Filtrar las columnas necesarias
df_num = df.select(numerical_columns + [target_column])

# Calcular la correlación entre las variables numéricas y la variable objetivo
correlations = {}
for col in numerical_columns:
    corr = df_num.select(col, target_column).stat.corr(col, target_column)
    correlations[col] = corr


# COMMAND ----------

# Convertir las correlaciones a un DataFrame de pandas
correlation_df = pd.DataFrame(list(correlations.items()), columns=['Feature', 'Correlation'])

# Ordenar el DataFrame por la correlación
correlation_df = correlation_df.sort_values(by='Correlation', ascending=False)

# Visualizar las correlaciones usando seaborn
plt.figure(figsize=(18, 9))
sns.barplot(data=correlation_df, x='Correlation', y='Feature', palette='viridis')
plt.axvline(x=0, color='red', linestyle='--')
plt.title('Correlación de los Features Númericos con Target')
plt.xlabel('Correlación')
plt.ylabel('Feature')
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC Las correlaciones entre diferentes variables y el ratio de ocupación de propiedades en Airbnb son generalmente bajas, indicando relaciones débiles. El precio es la variable más influyente, mientras que las calificaciones relacionadas con la experiencia del huésped y la capacidad de respuesta del anfitrión también juegan un papel importante. Las características con baja correlación, como el número de baños y la duración de las noches, parecen ser menos relevantes para este análisis específico.

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4. Matriz de Correlación
# MAGIC Para entender las relaciones entre las variables numéricas, se calculó y visualizó una matriz de correlación.

# COMMAND ----------

# Convertir el DataFrame de Spark a Pandas
pdf = df_num.toPandas()

# COMMAND ----------

# Calcular la matriz de correlaciones
corr_matrix = pdf.corr()

# Crear el heatmap
plt.figure(figsize=(22, 17))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)

# Mostrar el heatmap
plt.title('Heatmap de Correlaciones')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC El heatmap de correlaciones presentado muestra las relaciones entre diferentes variables del dataset. Se observa una fuerte correlación positiva entre `host_listings_count`, `host_total_listings_count` y otras variables relacionadas con el número de propiedades que un anfitrión tiene, indicando que un mayor número de propiedades listadas está asociado con un mayor `host_listings_count`. Asimismo, `minimum_minimum_nights`, `maximum_minimum_nights` y `minimum_nights` están altamente correlacionadas, lo cual es lógico dado que todas estas variables se refieren a las restricciones mínimas de noches. 
# MAGIC
# MAGIC Muchas variables no muestran una fuerte correlación entre sí, indicando que no existe una relación lineal directa significativa. Por ejemplo, `host_response_rate` no tiene una fuerte correlación con la mayoría de las otras variables. Las variables relacionadas con las calificaciones de las reseñas (`review_scores_rating`, `review_scores_accuracy`, `review_scores_cleanliness`, etc.) tienen correlaciones moderadas entre sí, lo que indica que buenas calificaciones en una área tienden a estar asociadas con buenas calificaciones en otras áreas. 
# MAGIC
# MAGIC Las variables de disponibilidad (`availability_30`, `availability_60`, `availability_90`, `availability_365`) muestran correlaciones entre ellas, indicando que la disponibilidad a corto plazo está relacionada con la disponibilidad a medio y largo plazo. Finalmente, la tasa de ocupación (`occupation_rate`) tiene correlaciones moderadas con algunas variables relacionadas con la disponibilidad y el número de reseñas, sugiriendo que a medida que la disponibilidad disminuye, la tasa de ocupación tiende a aumentar. Esto tiene sentido al ser una variable calculada a partir de al disponibilidad.
# MAGIC
# MAGIC En resumen, el heatmap de correlaciones proporciona una visión global de las relaciones entre variables, destacando áreas de fuerte correlación que pueden ser importantes para un análisis más detallado.

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5. Distribuciones
# MAGIC
# MAGIC Para analizar las distribuciones de las variables en nuestro dataset, se han creado gráficos de distribución utilizando histogramas y gráficos KDE (Kernel Density Estimation). Esto nos permite entender mejor cómo se distribuyen los datos para cada variable y detectar posibles sesgos o patrones.

# COMMAND ----------

# Variables a analizar
variables = ['price', 'bathrooms', 'bedrooms', 'beds', 'accommodates', 'minimum_nights', 'maximum_nights', 'availability_365', 'number_of_reviews', 'review_scores_rating']

plt.figure(figsize=(20, 20))
for i, var in enumerate(variables):
    plt.subplot(5, 2, i+1)
    var_df = df.select(var).toPandas()
    sns.histplot(var_df[var], kde=True, color=color, bins=30)
    plt.title(f'Distribución de {var}')
    plt.xlabel(var)
    plt.ylabel('Frecuencia')
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Las distribuciones de las variables en el dataset de Airbnb revelan varias tendencias importantes. La distribución de `price` muestra una caída pronunciada, con la mayoría de los precios concentrados por debajo de 100 dolares, sugiriendo que la mayoría de los listados son asequibles. Las variables `bathrooms`, `bedrooms` y `beds` muestran distribuciones sesgadas a la derecha, con la mayoría de los listados teniendo uno o dos baños, dormitorios y camas, respectivamente. La variable `accommodates` también está sesgada a la derecha, indicando que la mayoría de los listados pueden acomodar a entre una y cuatro personas.
# MAGIC
# MAGIC La distribución de `minimum_nights` está fuertemente sesgada hacia la izquierda, con la mayoría de los listados teniendo un mínimo de una a dos noches, mientras que `maximum_nights` tiene un valor atípico extremo, sugiriendo que algunos listados tienen requisitos de estancia máxima muy altos. `availability_365` muestra una distribución bimodal, indicando que algunos listados están disponibles durante todo el año, mientras que otros tienen disponibilidad limitada.
# MAGIC
# MAGIC La variable `number_of_reviews` muestra una alta concentración de listados con pocas reseñas, lo que puede indicar que muchos listados son relativamente nuevos o tienen poca actividad. Finalmente, `review_scores_rating` tiene una distribución sesgada hacia la derecha, con la mayoría de las puntuaciones de las reseñas concentradas en valores altos, lo que sugiere que la mayoría de los listados tienen buenas calificaciones.

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5. Test Estadisticos
# MAGIC Realizamos pruebas estadísticas para validar la independencia de variables categóricas y comprender mejor la estructura de los datos. El test de Chi-cuadrado se utiliza para determinar si existe una asociación significativa entre dos variables categóricas.

# COMMAND ----------

# Seleccionar las columnas categóricas relevantes
categoric_cols = ['host_is_superhost', 'host_has_profile_pic', 'host_identity_verified', 'neighbourhood_cleansed',
                  'property_type', 'room_type', 'bathrooms_text', 'has_availability', 'instant_bookable', 'city']

df_categorical = df.select(categoric_cols)

# COMMAND ----------

# Indexar variables categóricas
indexers = [StringIndexer(inputCol=column, outputCol=column + "_index") for column in categoric_cols]
pipeline = Pipeline(stages=indexers)
df_indexed = pipeline.fit(df_categorical).transform(df_categorical)

# Realizar el test de Chi-cuadrado para cada par de variables categóricas
results = []
for i in range(len(categoric_cols)):
    for j in range(i + 1, len(categoric_cols)):
        col1 = categoric_cols[i] + "_index"
        col2 = categoric_cols[j] + "_index"
        
        # Preparar los datos para el test de Chi-cuadrado
        df_chi = df_indexed.select(col(col1).cast('double'), col(col2).cast('double'))
        
        # Convertir a formato Vector
        assembler = VectorAssembler(inputCols=[col1], outputCol="features")
        df_vector = assembler.transform(df_chi).select('features', col(col2).alias('label'))
        
        # Realizar el test de Chi-cuadrado
        chi_square_test = ChiSquareTest.test(df_vector, 'features', 'label').head()
        
        # Guardar resultados
        results.append(Row(variable_1=categoric_cols[i],
                           variable_2=categoric_cols[j],
                           p_value=float(chi_square_test.pValues[0]),
                           degrees_of_freedom=int(chi_square_test.degreesOfFreedom[0]),
                           statistic=float(chi_square_test.statistics[0])))

# Crear DataFrame de resultados
results_df = spark.createDataFrame(results)

# COMMAND ----------

# Mostrar los resultados en formato de tabla
results_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC Los resultados muestran asociaciones significativas entre varias combinaciones de variables categóricas, indicando que las características de los anfitriones y las propiedades están interrelacionadas de manera compleja. Estas asociaciones pueden ayudar a entender mejor cómo ciertas características influyen en el comportamiento de los anfitriones y la oferta de propiedades en Airbnb. Los p-valores extremadamente bajos (próximos a 0) indican que es muy poco probable que estas asociaciones se deban al azar, reforzando la significancia de estos hallazgos.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Almacenar Base Preprocesada
# MAGIC Se guarda en un parquet la base preprocesada para que pueda utilizarse en el entrenamiento del modelo predictivo. 

# COMMAND ----------

# Muestra de la base
df.display()

# COMMAND ----------

print(f"El total de registros luego del preprocesamiento es: {df.count()}")

# COMMAND ----------

# Escribir base de datos preprocesada
df.write.mode("overwrite").parquet("/dbfs/FileStore/preprocessed_df.parquet")

# COMMAND ----------

