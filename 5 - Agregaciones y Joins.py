# Databricks notebook source
# MAGIC %sh curl https://raw.githubusercontent.com/databricks/Spark-The-Definitive-Guide/master/data/retail-data/all/online-retail-dataset.csv --output /tmp/online-retail-dataset.csv

# COMMAND ----------

dbutils.fs.mv("file:/tmp/online-retail-dataset.csv", "dbfs:/tmp/retail/online-retail-dataset.csv")

# COMMAND ----------

# MAGIC %fs
# MAGIC ls /tmp/

# COMMAND ----------

df = spark.read.format("csv")\
  .option("header", "true")\
  .option("inferSchema", "true")\
  .load("dbfs:/tmp/retail/*.csv")\
  .coalesce(5)
df.cache()
df.createOrReplaceTempView("dfTable")

# COMMAND ----------

# MAGIC %md
# MAGIC ##Count

# COMMAND ----------

from pyspark.sql.functions import count
df.select(count("StockCode")).show()

# COMMAND ----------

from pyspark.sql.functions import countDistinct
df.select(countDistinct("StockCode")).show() 

# COMMAND ----------

# MAGIC %md
# MAGIC ## First-Last

# COMMAND ----------

from pyspark.sql.functions import first, last
df.select(first("StockCode"), last("StockCode")).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Statistics

# COMMAND ----------

from pyspark.sql.functions import min, max
df.select(min("Quantity"), max("Quantity")).show()

# COMMAND ----------

from pyspark.sql.functions import sum
df.select(sum("Quantity")).show() 

# COMMAND ----------

from pyspark.sql.functions import sumDistinct
df.select(sumDistinct("Quantity")).show()

# COMMAND ----------

from pyspark.sql.functions import sum, count, avg, expr

df.select(
    count("Quantity").alias("total_transactions"),
    sum("Quantity").alias("total_purchases"),
    avg("Quantity").alias("avg_purchases"),
    expr("mean(Quantity)").alias("mean_purchases"))\
  .selectExpr(
    "total_transactions",
    "total_purchases",
    "total_purchases/total_transactions",
    "avg_purchases",
    "mean_purchases").show()

# COMMAND ----------

from pyspark.sql.functions import var_pop, stddev_pop
from pyspark.sql.functions import var_samp, stddev_samp
df.select(var_pop("Quantity"), var_samp("Quantity"),
  stddev_pop("Quantity"), stddev_samp("Quantity")).show()

# COMMAND ----------

from pyspark.sql.functions import skewness, kurtosis
df.select(skewness("UnitPrice"), kurtosis("UnitPrice")).show()

# COMMAND ----------

from pyspark.sql.functions import corr, covar_pop, covar_samp
df.select(corr("InvoiceNo", "Quantity"), covar_samp("InvoiceNo", "Quantity"),
    covar_pop("InvoiceNo", "Quantity")).show()

# COMMAND ----------

from pyspark.sql.functions import collect_set, collect_list
df.agg(collect_set("Country")).show(1, False)

# COMMAND ----------

df.agg(collect_set("Country"), collect_list("Country")).show()

# COMMAND ----------

# MAGIC %md
# MAGIC #Agregations

# COMMAND ----------

# MAGIC %md
# MAGIC ##Groupby

# COMMAND ----------

from pyspark.sql.functions import count

df.groupBy("InvoiceNo").agg(
    count("Quantity").alias("quan"), ##la mas utilizada
    expr("count(Quantity) as quan")).show()

# COMMAND ----------

df.groupBy("InvoiceNo", "Country").agg(expr("avg(Quantity)"),expr("stddev_pop(Quantity)"))\
  .show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Window

# COMMAND ----------

from pyspark.sql.functions import col, to_date
dfWithDate = df.withColumn("date", to_date(col("InvoiceDate"), "MM/d/yyyy H:mm"))
dfWithDate.createOrReplaceTempView("dfWithDate")

# COMMAND ----------

from pyspark.sql.window import Window
from pyspark.sql.functions import desc
windowSpec = Window\
  .partitionBy("CustomerId", "date")\
  .orderBy(desc("Quantity"))\
  .rowsBetween(Window.unboundedPreceding, Window.currentRow)

# COMMAND ----------

from pyspark.sql.functions import max
maxPurchaseQuantity = avg(col("Quantity")).over(windowSpec)

# COMMAND ----------

from pyspark.sql.functions import dense_rank, rank
purchaseDenseRank = dense_rank().over(windowSpec)
purchaseRank = rank().over(windowSpec)

# COMMAND ----------

spark.sql("set spark.sql.legacy.timeParserPolicy=LEGACY")

# COMMAND ----------

from pyspark.sql.functions import col

dfWithDate.where("CustomerId IS NOT NULL").orderBy("CustomerId")\
  .select(
    col("CustomerId"),
    col("date"),
    col("Quantity"),
    purchaseRank.alias("quantityRank"),
    purchaseDenseRank.alias("quantityDenseRank"),
    maxPurchaseQuantity.alias("maxPurchaseQuantity")).show(20)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Drop

# COMMAND ----------

dfNoNull = dfWithDate.drop()
dfNoNull.createOrReplaceTempView("dfNoNull")

# COMMAND ----------

dfNoNull.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Rollup

# COMMAND ----------

spark.sql("set spark.sql.legacy.timeParserPolicy=LEGACY")

# COMMAND ----------

dfNoNull.select(sum("Quantity")).show() 

# COMMAND ----------

rolledUpDF = dfNoNull.rollup("Date", "Country").agg(sum("Quantity"))\
  .selectExpr("Date", "Country", "`sum(Quantity)` as total_quantity")\
  .orderBy("Date")
rolledUpDF.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Cube

# COMMAND ----------

from pyspark.sql.functions import sum

display(dfNoNull.cube("Date", "Country").agg(sum(col("Quantity")))\
  .select("Date", "Country", "sum(Quantity)").orderBy("Date"))

# COMMAND ----------

# MAGIC %md
# MAGIC ##Pivot

# COMMAND ----------

pivoted = dfWithDate.groupBy("date").pivot("Country").sum()
pivoted.columns

# COMMAND ----------

display(pivoted.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC #Joins
# MAGIC https://spark.apache.org/docs/3.1.2/api/python/reference/api/pyspark.sql.DataFrame.join.html

# COMMAND ----------

person = spark.createDataFrame([
    (0, "Bill Chambers", 0, [100]),
    (1, "Matei Zaharia", 1, [500, 250, 100]),
    (2, "Michael Armbrust", 1, [250, 100])])\
  .toDF("id", "name", "graduate_program", "spark_status")
graduateProgram = spark.createDataFrame([
    (0, "Masters", "School of Information", "UC Berkeley"),
    (2, "Masters", "EECS", "UC Berkeley"),
    (1, "Ph.D.", "EECS", "UC Berkeley")])\
  .toDF("id", "degree", "department", "school")
sparkStatus = spark.createDataFrame([
    (500, "Vice President"),
    (250, "PMC Member"),
    (100, "Contributor")])\
  .toDF("id", "status")



# COMMAND ----------

person.show()
graduateProgram.show()
sparkStatus.show()

# COMMAND ----------

person.join(graduateProgram, person.graduate_program == graduateProgram['id'], 'inner').show()

# COMMAND ----------

person.join(graduateProgram, person.graduate_program == graduateProgram['id'], 'left').show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Union

# COMMAND ----------

gradProgram2 = graduateProgram.union(spark.createDataFrame([
    (0, "Masters", "Duplicated Row", "Duplicated School")]))

gradProgram2.createOrReplaceTempView("gradProgram2")

# COMMAND ----------

gradProgram2.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Join expr

# COMMAND ----------

from pyspark.sql.functions import expr

person.withColumnRenamed("id", "personId")\
  .join(sparkStatus, expr("array_contains(spark_status, id)")).show()