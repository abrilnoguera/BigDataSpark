# Databricks notebook source
from pyspark.sql.functions import *

# COMMAND ----------

# MAGIC %md
# MAGIC # Dataset Movies

# COMMAND ----------

import os

dbfs_dir = '/databricks-datasets/cs110x/ml-20m/data-001'
ratings_filename = dbfs_dir + '/ratings.csv' 
movies_filename = dbfs_dir + '/movies.csv'

# COMMAND ----------

movies = (spark.read.format("csv")
.options(header = True, inferSchema = True)
.load(movies_filename)
.cache()) # Keep the dataframe in memory for faster processing

# COMMAND ----------

ratings = (spark.read.format("csv")
.options(header = True, inferSchema = True)
.load(ratings_filename)
.cache())

# COMMAND ----------

movies.display()

# COMMAND ----------

ratings.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Genere un gráfico donde visualicemos la cantidad de películas por año
# MAGIC - Generar columna donde extraiga el año del título de la película
# MAGIC - Agrupar y contabilizar por año

# COMMAND ----------

movies = movies.withColumn("year", regexp_extract('title', r'\((\d{4})\)', 1))\
                .withColumn("year", when(col("year") == "", regexp_extract('title', r'\((\d{4})\-', 1)).otherwise(col("year")))\
                .withColumn("year", when(col("year") == "", regexp_extract('title', r'\((\d{4})\–', 1)).otherwise(col("year")))

# COMMAND ----------

# MAGIC %md
# MAGIC Hay años que no estan en los rangos del minimo al maximo por lo que se debe completar con 0.

# COMMAND ----------

movies_agg = movies.filter(col("year") != "").groupBy('year').agg(count("*").alias("Cantidad")).sort("year")

min_year = int(movies_agg.agg({"year": "min"}).collect()[0][0])
max_year = int(movies_agg.agg({"year": "max"}).collect()[0][0])

years = spark.range(min_year, max_year + 1).toDF("year")

years = years.join(movies_agg, ["year"], "left").na.fill(0)

years = years.orderBy("year")

# COMMAND ----------

import matplotlib.pyplot as plt
# Gráfico
df = years.toPandas()

plt.figure(figsize=(20, 8)) 
plt.plot(df['year'], df['Cantidad'], marker='o', linestyle='-', color='blue')
plt.title('Number of Movies by Year')
plt.xlabel('Year')
plt.ylabel('Number of Movies')
plt.grid(False)
plt.xticks([year for year in df['year'] if int(year) % 5 == 0])
plt.tight_layout() 
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Liste los géneros únicos existentes y cuantas peliculas hay en cada uno de ellos?
# MAGIC - Generar columna con el genero de la película, la pelicula se repetirá la cantidad de generos que tenga
# MAGIC - Agrupar y contabilizar por genero

# COMMAND ----------

 moviesExp = movies.withColumn("genre", split(col("genres"), "\|"))\
  .withColumn("genre", explode(col("genre")))

# COMMAND ----------

moviesExp.groupBy('genre')\
    .agg(count("*").alias("Cantidad"))\
    .display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Si existen nulos en el año de la pelicula, resuelva imputación
# MAGIC - Verifique si hay años que no se pudieron extraer y defina una metodología de imputación

# COMMAND ----------

# MAGIC %md
# MAGIC Hay 19 peliculas que no especifican el año o tienen mas de uno.

# COMMAND ----------

movies.filter(col("year") == "").display()

# COMMAND ----------

# MAGIC %md
# MAGIC Se inputa de la misma forma que se inputaron los generos: "(No year listed)"

# COMMAND ----------

movies = movies.na.replace([""], ["(No year listed)"], "year")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Cuales son las 3 películas con mayor promedio de rating con mas de 500 puntuaciones?
# MAGIC - Calcular el promedio de rating de cada pelicula
# MAGIC - Ordene y filtre las que tengan mas de 500 puntuaciones

# COMMAND ----------

ratings.groupBy("movieId").agg(avg("rating").alias("avg_rating"), count("*").alias("Cantidad"))\
    .filter(col("Cantidad") > 500)\
    .join(movies.select("movieId", "title"), ["movieId"], "left")\
    .orderBy(col("avg_rating").desc())\
    .display()

# COMMAND ----------

# MAGIC %md
# MAGIC Las 3 películas con mayor rating son "Shawshank Redemption", "Godfather" & "Usual Suspects".

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5. Genere un tabla pivot entre la decada de la pelicula y el genero del promedio de rating y plasmelo en un heatmap
# MAGIC - Calcular la decada de lanzamiento de la pelicula
# MAGIC - Generar una tabla pivot con el promedio de rating entre el año y su genero
# MAGIC - Graficar el heatmap

# COMMAND ----------

movies_ratings = moviesExp.filter(col("year") != "")\
                    .filter(col("genre")!="(no genres listed)")\
                    .withColumn("decade", format_string("%d0s", (col("year") / 10).cast("int")))

# COMMAND ----------

movies_ratings = movies_ratings.join(ratings, ["movieId"], "left")

# COMMAND ----------

df = movies_ratings.groupBy("genre")\
            .pivot("decade")\
            .agg(round(avg("rating"),2).alias("avg_rating"))\
            .fillna(0)\
            .toPandas()

# COMMAND ----------

import seaborn as sns

# Crear un heatmap usando seaborn
plt.figure(figsize=(16, 8))  
heatmap = sns.heatmap(df.set_index('genre'), annot=True, fmt=".2f", cmap="YlGnBu")

# Añadir títulos y etiquetas según sea necesario
plt.title('Average Rating by Genre and Decade')
plt.xlabel('Decade')
plt.ylabel('Genre')

# Asegurar que las etiquetas de los ejes no se superpongan
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()

# Mostrar el gráfico
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Dataset stocks

# COMMAND ----------

# MAGIC %sh curl https://raw.githubusercontent.com/nikoloide/itba_data/main/prices.csv --output /tmp/prices.csv

# COMMAND ----------

# MAGIC %sh curl https://raw.githubusercontent.com/nikoloide/itba_data/main/securities.csv --output /tmp/securities.csv

# COMMAND ----------

dbutils.fs.mv("file:/tmp/prices.csv", "dbfs:/tmp/stock/prices.csv")

# COMMAND ----------

dbutils.fs.mv("file:/tmp/securities.csv", "dbfs:/tmp/stock/securities.csv")

# COMMAND ----------

prices = (spark.read.format("csv")
    .options(header = True, inferSchema = True)
    .load("/tmp/stock/prices.csv"))

securities = (spark.read.format("csv")
.options(header = True, inferSchema = True)
.load("/tmp/stock/securities.csv"))       

# COMMAND ----------

prices.show(5)
securities.show(5,False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Genere un promedio móvil de un marco de 20 días de cada una de las acciones
# MAGIC - Consideramos precio de la acción al valor de cierre del título en el día

# COMMAND ----------

from pyspark.sql.window import Window

# Definir la ventana de partición por 'action' y ordenar por 'date'
windowSpec = Window.partitionBy('symbol').orderBy('date').rowsBetween(-19, 0)

# Calcular el promedio móvil de 20 días para cada acción y cada fecha
moving_averages = prices.withColumn('20_day_moving_avg', avg(col('close')).over(windowSpec))

# Mostrar los resultados
moving_averages.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Genere un gráfico de histograma comparando la cotización entre dos tipo de industria
# MAGIC - Primero deberá joinear el dataset prices con securities para obtener el tipo de industria por acción
# MAGIC - Seleccione dos industrias y genere un gráfico comparativo de histograma

# COMMAND ----------

prices_securities = prices.join(securities, prices.symbol == securities["`Ticker symbol`"], "left")

# COMMAND ----------

prices_securities.filter((col("GICS Sub Industry") == "IT Consulting & Other Services") | (col("GICS Sub Industry") == "Home Entertainment Software"))\
        .select("close", "GICS Sub Industry")\
        .toPandas()\
        .hist(by='GICS Sub Industry',figsize = (24,5))

# COMMAND ----------

prices_securities_hist = prices_securities.filter((col("GICS Sub Industry") == "IT Consulting & Other Services") | (col("GICS Sub Industry") == "Home Entertainment Software"))\
        .select("close", "GICS Sub Industry")\
        .toPandas()

fig, ax = plt.subplots(figsize=(20, 8))
sns.histplot(prices_securities_hist, bins=10, x="close", hue="GICS Sub Industry")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Mida la correlación entre la cotización de las acciones de una industria y genere un gráfico de matriz de correlación
# MAGIC - Seleccione un grupo de acciones de una determinada industria y mida la correlación entre ellas

# COMMAND ----------

prices_corr = prices_securities.filter(col("GICS Sub Industry") == "IT Consulting & Other Services")

# COMMAND ----------

prices_corr = prices_corr.groupBy("date").pivot("symbol").agg(first("close")).fillna(0)

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation

assembler = VectorAssembler(inputCols=prices_corr.columns[1:], outputCol="features")
df_vector = assembler.transform(prices_corr).select("features")

matrix = Correlation.corr(df_vector, "features").head()

# COMMAND ----------

import numpy as np

correlation_matrix = matrix[0].toArray()

# COMMAND ----------

plt.figure(figsize=(20, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=prices_corr.columns[1:], yticklabels=prices_corr.columns[1:])
plt.title("Matriz de Correlación de las Acciones de Tecnología")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Elija uno de los títulos y genere un gráfico visualizando el promedio móvil generado en el punto 1 y la variación diaria de la acción
# MAGIC - Primero calcular la variación diaria del precio de cierre de stock
# MAGIC - Generar el gráfico combinado de lineas y barras

# COMMAND ----------

# Suponiendo que quieres calcular esto para una acción específica, por ejemplo, 'AAPL'
df_aapl = prices.filter(col("symbol") == "AAPL").orderBy("date")

# Crear ventana para obtener el valor previo
windowSpec = Window.orderBy("date")

# Calcular la variación diaria
df_aapl = df_aapl.withColumn("previous_close", lag("close", 1).over(windowSpec))
df_aapl = df_aapl.withColumn("daily_change", 
                             round((col("close") - col("previous_close")) / col("previous_close") * 100, 2))\
                        .fillna(0, ["previous_close", "daily_change"])

# Join con promedio movil
df_aapl = df_aapl.join(moving_averages.select("symbol", "date", "20_day_moving_avg"), ["symbol", "date"], "left")

df_aapl.display()

# COMMAND ----------

df = df_aapl.select("date", "close", "daily_change", "20_day_moving_avg").toPandas()

fig, ax1 = plt.subplots(figsize=(20, 10))

# Gráfico de línea para el promedio móvil
ax1.plot(df['date'], df['20_day_moving_avg'], color='b', label='20-Day Moving Average')
ax1.set_xlabel('Date')
ax1.set_ylabel('Close Price', color='b')
ax1.tick_params(axis='y', labelcolor='b')

# Crear un segundo eje para la variación diaria
ax2 = ax1.twinx()
ax2.bar(df['date'], df['daily_change'], color='r', label='Daily Change', alpha=0.3)
ax2.set_ylabel('Daily Change (%)', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# Títulos y leyendas
ax1.set_title('20-Day Moving Average and Daily Change of AAPL Stock')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5. Detecta algun valor atípico en la serie para alguno de los títulos? Identifique la fecha, acción y precio
# MAGIC - Definir una metodología para detectar valores atípicos de la serie diaria de los títulos y obtener en que fecha fue, su precio y que variación tuvo respecto al día anterior

# COMMAND ----------

# Calcular la media y la desviación estándar del precio de cierre agrupados por 'symbol'
stats = prices.groupBy("symbol").agg(
    mean("close").alias("mean"),
    stddev("close").alias("stddev")
)

# COMMAND ----------

# Crear ventana para obtener el valor previo
windowSpec = Window.partitionBy('symbol').orderBy("date")

# Calcular la variación diaria
prices = prices.withColumn("previous_close", lag("close", 1).over(windowSpec))
prices = prices.withColumn("daily_change", 
                             round((col("close") - col("previous_close")) / col("previous_close") * 100, 2))\
                        .fillna(0, ["previous_close", "daily_change"])

# COMMAND ----------

prices = prices.join(stats, "symbol")

# COMMAND ----------

# Identificar outliers
df_outliers = prices.withColumn("is_outlier", 
                                   (col("close") < (col("mean") - 3 * col("stddev"))) | 
                                   (col("close") > (col("mean") + 3 * col("stddev"))))

# COMMAND ----------

df_outliers = df_outliers.filter("is_outlier").select("date", "symbol", "close", "daily_change").orderBy("daily_change")
df_outliers.display()


# COMMAND ----------

