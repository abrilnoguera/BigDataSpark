# Databricks notebook source
import pandas as pd

# COMMAND ----------

from pyspark.sql.functions import *

# COMMAND ----------

display(dbutils.fs.ls('/databricks-datasets/credit-card-fraud'))

# COMMAND ----------

display(dbutils.fs.ls('/databricks-datasets/wine-quality/'))

# COMMAND ----------

df_red = spark.read.format("csv")\
    .option("sep", ";")\
  .option("header", "true")\
  .option("inferSchema", "true")\
  .load("dbfs:/databricks-datasets/wine-quality/winequality-red.csv")\
  .coalesce(5)

# COMMAND ----------

df_red.show(5)

# COMMAND ----------

df_white = spark.read.format("csv")\
    .option("sep", ";")\
  .option("header", "true")\
  .option("inferSchema", "true")\
  .load("dbfs:/databricks-datasets/wine-quality/winequality-white.csv")\
  .coalesce(5)

# COMMAND ----------

df_white.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Union dataset

# COMMAND ----------

df_red = df_red.withColumn("type", lit('RED'))
df_white = df_white.withColumn("type", lit('WHITE'))

# COMMAND ----------

df_wine = df_red.union(df_white)

# COMMAND ----------

display(df_wine)

# COMMAND ----------

df_wine.groupBy('type').count().show()

# COMMAND ----------

df_wine.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Missing Values

# COMMAND ----------

df_wine = df_wine.replace({1.6:None}, subset=['residual sugar'])

# COMMAND ----------

from pyspark.sql.functions import col,isnan, when, count
df_wine.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df_wine.columns]
   ).show(vertical=True)

# COMMAND ----------

from pyspark.sql.types import StringType, DoubleType

categoric_cols = [f.name for f in df_wine.schema.fields if isinstance(f.dataType, StringType)]
numeric_cols = [f.name for f in df_wine.schema.fields if isinstance(f.dataType, DoubleType)]

# COMMAND ----------

categoric_cols

# COMMAND ----------

numeric_cols

# COMMAND ----------

from pyspark.ml.feature import Imputer
#https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.Imputer.html
imputer = Imputer(
    inputCols=numeric_cols, 
    strategy='mean',
    outputCols=["{}".format(c) for c in numeric_cols]
)
df_wine_tf = imputer.fit(df_wine).transform(df_wine)

# COMMAND ----------

df_wine_tf.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df_wine_tf.columns]
   ).show(vertical=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Atypical Data

# COMMAND ----------

import matplotlib.pyplot as plt
df_plot = df_wine.select(numeric_cols).toPandas()
df_plot.plot(kind='box', subplots=True, layout=(3,4), sharex=False, sharey=False,figsize=(20, 10))
fig = plt.show()
display(fig)

# COMMAND ----------

from pyspark.sql.functions import col, exp
def iqr_outlier_treatment(dataframe, columns, factor=1.5):
    """
    Detects and treats outliers using IQR for multiple variables in a PySpark DataFrame.

    :param dataframe: The input PySpark DataFrame
    :param columns: A list of columns to apply IQR outlier treatment
    :param factor: The IQR factor to use for detecting outliers (default is 1.5)
    :return: The processed DataFrame with outliers treated
    """
    for column in columns:
        # Calculate Q1, Q3, and IQR
        quantiles = dataframe.approxQuantile(column, [0.25, 0.75], 0.01)
        q1, q3 = quantiles[0], quantiles[1]
        iqr = q3 - q1

        # Define the upper and lower bounds for outliers
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr

        # Filter outliers and update the DataFrame
        dataframe = dataframe.filter((col(column) >= lower_bound) & (col(column) <= upper_bound))

    return dataframe

# COMMAND ----------

df_outlier_treatment = iqr_outlier_treatment(df_wine, numeric_cols, factor=1.5)

# COMMAND ----------

df_wine.count()
df_outlier_treatment.count()

# COMMAND ----------

df_plot = df_outlier_treatment.select(numeric_cols).toPandas()
df_plot.plot(kind='box', subplots=True, layout=(3,4), sharex=False, sharey=False,figsize=(20, 10))
fig = plt.show()
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Correlation

# COMMAND ----------

from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler

# convert to vector column first
vector_col = "corr_features"
assembler = VectorAssembler(inputCols=numeric_cols, outputCol=vector_col)
df_vector = assembler.transform(df_wine_tf).select(vector_col)

# get correlation matrix
matrix = Correlation.corr(df_vector, vector_col)

# COMMAND ----------

display(df_vector)

# COMMAND ----------

matrix.collect()[0]["pearson({})".format(vector_col)].values

# COMMAND ----------

matrix = Correlation.corr(df_vector, 'corr_features').collect()[0][0] 
corr_matrix = matrix.toArray().tolist() 
corr_matrix_df = pd.DataFrame(data=corr_matrix, columns = numeric_cols, index=numeric_cols) 

import seaborn as sns 
import matplotlib.pyplot as plt

plt.figure(figsize=(16,5))  
sns.heatmap(corr_matrix_df, 
            xticklabels=corr_matrix_df.columns.values,
            yticklabels=corr_matrix_df.columns.values,  cmap="vlag_r", annot=True)

#https://seaborn.pydata.org/tutorial/color_palettes.html


# COMMAND ----------

corr_matrix_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Analysis

# COMMAND ----------

df_wine_tf.select('fixed acidity').toPandas().hist()

# COMMAND ----------

df_wine_tf.select('fixed acidity', 'type').toPandas().hist(by='type',figsize = (10,5))

# COMMAND ----------

import seaborn as sns
df_wine_tf_hist = df_wine_tf.select('fixed acidity', 'type').toPandas()
fig, ax = plt.subplots(figsize=(11, 8))
sns.histplot(df_wine_tf_hist, bins=10, x="fixed acidity", hue="type")

# COMMAND ----------

df_wine_tf_hist = df_wine_tf.select('fixed acidity','density', 'type').toPandas()
fig, ax = plt.subplots(figsize=(11, 8))
sns.histplot(df_wine_tf_hist, x="fixed acidity", y="density", hue="type")

# COMMAND ----------

rdd_wine = df_wine_tf.select(numeric_cols).rdd.flatMap(lambda x:x)
rdd_wine = rdd_wine.histogram(10)
rdd_hist = rdd_wine.toDF()

# COMMAND ----------

import numpy as np

#rdd_wine = df_wine_tf.select(numeric_cols).rdd.flatMap(lambda x:x)
#rdd_wine = rdd_wine.histogram(10)

plot_data = df_wine_tf.select('fixed acidity').toPandas()
x= plot_data['fixed acidity']

hist, bin_edges = np.histogram(x,10,weights=np.zeros_like(x) + 100. / x.size) # make the histogram

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1)
# Plot the histogram heights against integers on the x axis
ax.bar(range(len(hist)),hist,width=1,alpha=0.8,ec ='black',color = 'gold')

# COMMAND ----------

#fig, ax = plt.subplots(4, 3, sharex='col', sharey='row', figsize=(20, 18))
n = 3
n_bins = 10
fig = plt.figure(figsize=(15, 8))
axs = fig.subplots(2, 6)
for c,i in zip(numeric_cols, range(len(numeric_cols))):
    plot_data = df_wine_tf.select(c).toPandas()
    x= plot_data[c]
    hist, bin_edges = np.histogram(x,10,weights=np.zeros_like(x) + 100. / x.size) # make the histogram
    #fig = plt.figure(figsize=(4, 3))
    #ax = fig.add_subplot(1, i, 1)
    # Plot the histogram heights against integers on the x axis
    if i<6:
        axs[0, i].bar(range(len(hist)),hist, width=1,alpha=0.8,ec ='black',color = 'gold')
        axs[0, i].set_title(c)
    else:
        axs[1, i-6].bar(range(len(hist)),hist, width=1,alpha=0.8,ec ='black',color = 'gold')
        axs[1, i-6].set_title(c)

# COMMAND ----------

from pyspark.ml.linalg import Vectors
from pyspark.ml.stat import ChiSquareTest

data = [(0.0, Vectors.dense(0.5, 10.0)),
        (0.0, Vectors.dense(1.5, 20.0)),
        (1.0, Vectors.dense(1.5, 30.0)),
        (0.0, Vectors.dense(3.5, 30.0)),
        (0.0, Vectors.dense(3.5, 40.0)),
        (1.0, Vectors.dense(3.5, 40.0))]
df = spark.createDataFrame(data, ["label", "features"])

r = ChiSquareTest.test(df_vector, "features", "label").head()
print("pValues: " + str(r.pValues))
print("degreesOfFreedom: " + str(r.degreesOfFreedom))
print("statistics: " + str(r.statistics))

# COMMAND ----------

from pyspark.ml.feature import StringIndexer
indexer = StringIndexer(inputCol="type", outputCol="type_i")
df_indexed = indexer.fit(df_wine_tf).transform(df_wine_tf)

# COMMAND ----------

vector_col = "corr_features"
assembler = VectorAssembler(inputCols=numeric_cols, outputCol=vector_col)
df_vector = assembler.transform(df_indexed).select("type_i", vector_col)

# COMMAND ----------

df_vector.take(5)

# COMMAND ----------

from pyspark.ml.stat import ChiSquareTest
r = ChiSquareTest.test(df_vector, vector_col, "type_i").head()
print("pValues: " + str(r.pValues))
print("degreesOfFreedom: " + str(r.degreesOfFreedom))
print("statistics: " + str(r.statistics))

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.types import *

chi = ChiSquareTest.test(df_vector, vector_col, "type_i")
temp = chi.rdd.map(lambda x:[float(y) for y in x['statistics']]).toDF(numeric_cols)

temp.show(vertical=True)

# COMMAND ----------

df_wine_tf_hist = df_outlier_treatment.select('chlorides', 'type').toPandas()
fig, ax = plt.subplots(figsize=(11, 8))
sns.histplot(df_wine_tf_hist, bins=50, x="chlorides", hue="type", element="step")