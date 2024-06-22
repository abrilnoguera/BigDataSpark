# Databricks notebook source
# MAGIC %sh curl https://raw.githubusercontent.com/databricks/Spark-The-Definitive-Guide/master/data/retail-data/by-day/2010-12-01.csv --output /tmp/2010-12-01.csv

# COMMAND ----------

dbutils.fs.mv("file:/tmp/2010-12-01.csv", "dbfs:/tmp/2010-12-01.csv")

# COMMAND ----------

# MAGIC %fs
# MAGIC ls dbfs:/tmp/2010-12-01.csv

# COMMAND ----------

df = spark.read.format("csv")\
  .option("header", "true")\
  .option("inferSchema", "true")\
  .load("/tmp/2010-12-01.csv")
df.printSchema()
df.createOrReplaceTempView("dfTable")

# COMMAND ----------

df.show(5, False)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Lit
# MAGIC Se utiliza para agregar una nueva columna a DataFrame asignando un valor literal o constante.
# MAGIC https://spark.apache.org/docs/3.1.3/api/python/reference/api/pyspark.sql.functions.lit.html

# COMMAND ----------

from pyspark.sql.functions import lit
df.select(lit(5), lit("five"), lit(5.0))

# COMMAND ----------

# MAGIC %md
# MAGIC ##Col
# MAGIC Proporciona varias funciones para trabajar con DataFrame para manipular los valores de las columnas, evaluar la expresión booleana para filtrar filas, recuperar un valor o parte de un valor de una columna de DataFrame \
# MAGIC https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.functions.col.html

# COMMAND ----------

from pyspark.sql.functions import col
df.where(col("InvoiceNo") != 536365)\
  .select("InvoiceNo", "Description")\
  .show(5, False)

# COMMAND ----------

# MAGIC %md
# MAGIC ##instr
# MAGIC Localice la posición de la primera aparición de la columna substr en la cadena dada. Devuelve nulo si alguno de los argumentos es nulo.
# MAGIC https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.functions.instr.html

# COMMAND ----------

from pyspark.sql.functions import instr
priceFilter = col("UnitPrice") > 600
descripFilter = instr(df.Description, "POSTAGE") >= 1
df.where(df.StockCode.isin("DOT")).where(priceFilter | descripFilter).show()

# COMMAND ----------

from pyspark.sql.functions import instr
DOTCodeFilter = col("StockCode") == "DOT"
priceFilter = col("UnitPrice") > 600
descripFilter = instr(col("Description"), "POSTAGE") >= 1
df.withColumn("isExpensive", DOTCodeFilter & (priceFilter | descripFilter))\
  .where("isExpensive")\
  .select("unitPrice", "isExpensive").show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ##expr
# MAGIC Analiza la cadena de expresión en la columna que representa
# MAGIC https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.functions.expr.html

# COMMAND ----------

from pyspark.sql.functions import expr
df.withColumn("isExpensive", expr("NOT UnitPrice <= 250"))\
  .where("isExpensive")\
  .select("Description", "UnitPrice", "isExpensive").show(5)

# COMMAND ----------

from pyspark.sql.functions import expr, pow
fabricatedQuantity = pow(col("Quantity") * col("UnitPrice"), 2) + 5
df.select(expr("CustomerId"), fabricatedQuantity.alias("realQuantity")).show(2)

# COMMAND ----------

# MAGIC %md
# MAGIC ##selectExpr
# MAGIC Proyecta un conjunto de expresiones SQL y devuelve un nuevo DataFrame.\
# MAGIC https://spark.apache.org/docs/3.1.3/api/python/reference/api/pyspark.sql.DataFrame.selectExpr.html

# COMMAND ----------

df.selectExpr(
  "CustomerId",
  "(POWER((Quantity * UnitPrice), 2.0) + 5) as realQuantity").show(2)

# COMMAND ----------

from pyspark.sql.functions import lit, round, bround

df.select(round(lit("2.5")), round(lit("2.5"),1), bround(lit("2.5"))).show(2)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Correlation

# COMMAND ----------

from pyspark.sql.functions import corr
df.stat.corr("Quantity", "UnitPrice")

# COMMAND ----------

df.select(corr("Quantity", "UnitPrice")).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Describe

# COMMAND ----------

df.describe().show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Estadisticos
# MAGIC Calcula los cuantiles aproximados de las columnas numéricas de un DataFrame.\
# MAGIC https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.approxQuantile.html

# COMMAND ----------

from pyspark.sql.functions import count, mean, stddev_pop, min, max

colName = "UnitPrice"
quantileProbs = [0.5]
relError = 0.05
df.stat.approxQuantile("UnitPrice", quantileProbs, relError) 

# COMMAND ----------

df.show()

# COMMAND ----------

df.stat.crosstab("StockCode", "Country").show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ##freqitem
# MAGIC Encontrar elementos frecuentes para las columnas\
# MAGIC https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.freqItems.html

# COMMAND ----------

df.stat.freqItems(["Country"]).show(1,False)

# COMMAND ----------

display(df.stat.freqItems(["StockCode", "Quantity"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ##monotonically_increasing_id
# MAGIC https://spark.apache.org/docs/3.1.3/api/python/reference/api/pyspark.sql.functions.monotonically_increasing_id.html

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id
df.select(monotonically_increasing_id()).show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ##initcap, lower, upper

# COMMAND ----------

from pyspark.sql.functions import initcap
df.select(col("Description"), initcap(col("Description"))).show()

# COMMAND ----------

from pyspark.sql.functions import lower, upper
df.select(col("Description"),
    lower(col("Description")),
    upper(lower(col("Description")))).show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ##lit, ltrim, rtrim, rpad, lpad, trim

# COMMAND ----------

from pyspark.sql.functions import lit, ltrim, rtrim, rpad, lpad, trim
df.select(
    ltrim(lit("    HELLO    ")).alias("ltrim"),
    rtrim(lit("    HELLO    ")).alias("rtrim"),
    trim(lit("    HELLO    ")).alias("trim"),
    lpad(lit("HELLO"), 10, "*").alias("lp"),
    rpad(lit("HELLO"), 10, "*").alias("rp")).show(2)

# COMMAND ----------

# MAGIC %md
# MAGIC ##regexp

# COMMAND ----------

from pyspark.sql.functions import regexp_replace
regex_string = "BLACK|WHITE|RED|GREEN|BLUE"
df.select(
  regexp_replace(col("Description"), regex_string, "COLOR").alias("color_clean"),
  col("Description")).show(5, False)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Translate
# MAGIC Una función traduce cualquier carácter en srcCol por un carácter coincidente. Los caracteres en reemplazo corresponden a los caracteres en coincidencia. La traducción se producirá cuando cualquier carácter de la cadena coincida con el carácter de la coincidencia.\
# MAGIC https://spark.apache.org/docs/3.1.1/api/python/reference/api/pyspark.sql.functions.translate.html

# COMMAND ----------

from pyspark.sql.functions import translate
df.select(translate(col("Description"), "LEET", "1337"),col("Description"))\
  .show(5, False)

# COMMAND ----------

from pyspark.sql.functions import regexp_extract
extract_str = "(BLACK|WHITE|RED|GREEN|BLUE)"
df.select(
     regexp_extract(col("Description"), extract_str, 1).alias("color_clean"),
     col("Description")).show(5, False)

# COMMAND ----------

from pyspark.sql.functions import instr
containsBlack = instr(col("Description"), "BLACK") >= 1
containsWhite = instr(col("Description"), "WHITE") >= 1
df.withColumn("hasSimpleColor", containsBlack | containsWhite)\
  .where("hasSimpleColor")\
  .select("Description", "hasSimpleColor").show(5, False)

# COMMAND ----------

from pyspark.sql.functions import expr, locate

simpleColors = ["black", "white", "red", "green", "blue"]

def color_locator(column, color_string):
  return locate(color_string.upper(), column)\
          .cast("boolean")\
          .alias("is_" + color_string)

selectedColumns = [color_locator(df.Description, c) for c in simpleColors]
selectedColumns.append(expr("*"))

df.select(*selectedColumns).where(expr("is_white OR is_red"))\
  .select("Description", "is_white", "is_red").show(3, False)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Dates

# COMMAND ----------

from pyspark.sql.functions import current_date, current_timestamp
dateDF = spark.range(10)\
  .withColumn("today", current_date())\
  .withColumn("now", current_timestamp())
dateDF.createOrReplaceTempView("dateTable")

# COMMAND ----------

dateDF.show(10,False)

# COMMAND ----------

from pyspark.sql.functions import date_add, date_sub
dateDF.select(date_sub(col("today"), 5), date_add(col("today"), 5)).show(1)

# COMMAND ----------

from pyspark.sql.functions import datediff, months_between, to_date
dateDF.withColumn("week_ago", date_sub(col("today"), 7))\
  .select("week_ago", "today", datediff(col("week_ago"), col("today"))).show(1)

dateDF.select(
    to_date(lit("2016-01-01")).alias("start"),
    to_date(lit("2017-05-22")).alias("end"))\
  .select(months_between(col("start"), col("end"))).show(1)


# COMMAND ----------

# MAGIC %md
# MAGIC ##toDate
# MAGIC Converts a Column into pyspark.sql.types.DateType using the optionally specified format.
# MAGIC https://spark.apache.org/docs/3.1.2/api/python/reference/api/pyspark.sql.functions.to_date.html

# COMMAND ----------

from pyspark.sql.functions import to_date, lit, date_format
spark.range(5).withColumn("date", lit("2017-01-01"))\
  .select(to_date(col("date")), date_format(to_date(col("date")), 'dd/MM/yyy'), date_format(to_date(col("date")), 'yyyyMM')).show(1)


# COMMAND ----------

from pyspark.sql.functions import to_date
dateFormat = "yyyy-dd-MM"
cleanDateDF = spark.range(1).select(
    to_date(lit("2017-12-11"), dateFormat).alias("date"),
    to_date(lit("2017-20-12"), dateFormat).alias("date2"))
cleanDateDF.createOrReplaceTempView("dateTable2")

# COMMAND ----------

from pyspark.sql.functions import to_timestamp
cleanDateDF.select(to_timestamp(col("date"), dateFormat)).show()


# COMMAND ----------

from pyspark.sql.functions import coalesce
df.select(col("Description"), col("CustomerId"), coalesce(col("Description"), col("CustomerId"))).show(5, False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Nulos

# COMMAND ----------

df.count()

# COMMAND ----------

df.na.drop("all", subset=["CustomerId"]).count()

# COMMAND ----------

#https://spark.apache.org/docs/3.1.2/api/python/reference/api/pyspark.sql.DataFrameNaFunctions.fill.html
df.na.fill("all", subset=["StockCode", "InvoiceNo"])

# COMMAND ----------

fill_cols_vals = {"StockCode": 5, "Description" : "No Value"}
df.na.fill(fill_cols_vals).show(5,False)


# COMMAND ----------

df.na.replace([""], ["UNKNOWN"], "Description").show(5,False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Complex

# COMMAND ----------

from pyspark.sql.functions import struct
complexDF = df.select(struct("Description", "InvoiceNo").alias("complex"))
complexDF.createOrReplaceTempView("complexDF")

# COMMAND ----------

complexDF.show(5, False)

# COMMAND ----------


from pyspark.sql.functions import split
df.select(split(col("Description"), " ")).show(5, False)

# COMMAND ----------

df.select(split(col("Description"), " ").alias("array_col"))\
  .selectExpr("array_col[0]").show(5)

# COMMAND ----------

from pyspark.sql.functions import size
df.select(col("Description"), size(split(col("Description"), " "))).show(5, False)

# COMMAND ----------

from pyspark.sql.functions import array_contains
df.select(col("Description"),array_contains(split(col("Description"), " "), "WHITE")).show(5, False)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Explode
# MAGIC Devuelve una nueva fila para cada elemento de la matriz o mapa dado. Utiliza el nombre de columna predeterminado col para los elementos de la matriz y la clave y el valor para los elementos del mapa, a menos que se especifique lo contrario.\
# MAGIC https://spark.apache.org/docs/3.1.3/api/python/reference/api/pyspark.sql.functions.explode.html

# COMMAND ----------

from pyspark.sql.functions import split, explode

df.withColumn("splitted", split(col("Description"), " "))\
  .withColumn("exploded", explode(col("splitted")))\
  .select("Description", "InvoiceNo", "exploded").show(10, False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Map

# COMMAND ----------

from pyspark.sql.functions import create_map
df.select(create_map(col("Description"), col("InvoiceNo")).alias("complex_map"))\
  .show(5, False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## UDF

# COMMAND ----------

udfExampleDF = spark.range(5).toDF("num")
def power3(double_value):
  return double_value ** 3
power3(2.0)

# COMMAND ----------

from pyspark.sql.functions import col
udfExampleDF.select(power3(col("num"))).show(2)