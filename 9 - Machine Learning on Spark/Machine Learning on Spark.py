# Databricks notebook source
# MAGIC %md-sandbox # ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Spark ML and Distributed Computing
# MAGIC
# MAGIC * Introduce machine learning
# MAGIC * Explore machine learning on Spark
# MAGIC * Import a notebook and create a cluster
# MAGIC
# MAGIC <img src="http://insideairbnb.com/images/insideairbnb_graphic_site_1200px.png" style="width:800px"/>
# MAGIC

# COMMAND ----------

# MAGIC %md ## What is machine learning?<br>
# MAGIC
# MAGIC * A diverse set of tools for understanding data
# MAGIC * Learns from data without being explicitly programmed
# MAGIC * Use cases include...
# MAGIC   - **Fraud Detection** 
# MAGIC   - **A/B Testing**
# MAGIC   - **Image Recognition** 
# MAGIC   - **Natural Language Processing**
# MAGIC   - **Financial Forecasting**
# MAGIC   - **Churn Analysis**
# MAGIC

# COMMAND ----------

# MAGIC %md-sandbox ## Why Spark machine learning?<br>
# MAGIC
# MAGIC * Scale
# MAGIC   - Process more data than can fit in any one machine
# MAGIC   - More data == performant models
# MAGIC * Works with pre-existing pipelines and tools
# MAGIC   - Spark (streaming, ETL, ad hoc analysis, reporting)
# MAGIC   - Frameworks (Spark ML, sklearn, Tensorflow and Horovod, R)
# MAGIC   - Languages (Python, R, Scala, SQL, Java)
# MAGIC * Model training _and_ production model serving
# MAGIC

# COMMAND ----------

# MAGIC %md ## Starting a ML Cluster 

# COMMAND ----------

# MAGIC %md-sandbox # ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Data Import and Exploratory Analysis
# MAGIC
# MAGIC ## In this video:<br>
# MAGIC
# MAGIC * Import an AirBnB Dataset
# MAGIC * Explore the dataset
# MAGIC * Visualize the data
# MAGIC
# MAGIC <div><img src="https://files.training.databricks.com/images/eLearning/ML-Part-1/eda.png" style="height: 500px; margin: 20px"/></div>
# MAGIC

# COMMAND ----------

# MAGIC %md-sandbox ## Data Import
# MAGIC
# MAGIC Spark connects to a wide variety of data sources including:  
# MAGIC <br>
# MAGIC * Traditional databases like Postgres, SQL Server, and MySQL
# MAGIC * Message brokers like Kafka and Kinesis
# MAGIC * Distributed databases like Cassandra and Redshift
# MAGIC * Data warehouses like Hive
# MAGIC * File types like CSV, Parquet, and Avro
# MAGIC
# MAGIC <div><img src="https://files.training.databricks.com/images/eLearning/ETL-Part-1/Workload_Tools_2-01.png" style="height: 500px; margin: 20px"/></div>

# COMMAND ----------

# MAGIC %run ./Mount-Datasets

# COMMAND ----------

filePath = "/mnt/training/airbnb/sf-listings/sf-listings-clean.parquet"

airbnbDF = spark.read.parquet(filePath)

# COMMAND ----------

# MAGIC %md ## Data Exploration<br>
# MAGIC
# MAGIC 1. What do each of the variables mean?
# MAGIC 2. Are there missing values?
# MAGIC 3. How many records are in the dataset?
# MAGIC 4. What is the mean and standard deviation of the dataset?
# MAGIC 5. Which variables are continuous and which are categorical?

# COMMAND ----------

airbnbDF.dtypes

# COMMAND ----------

display(airbnbDF)

# COMMAND ----------

display(airbnbDF.describe())

# COMMAND ----------

# MAGIC %md ## Data Visualization

# COMMAND ----------

display(airbnbDF.select("price"))

# COMMAND ----------

from pyspark.sql.functions import log

airbnbDF = airbnbDF.withColumn("logPrice", log("price"))

display(airbnbDF.select("logPrice"))

# COMMAND ----------

cols = [
  "logPrice",
  "host_total_listings_count",
  "review_scores_rating",
  "number_of_reviews",
  "bathrooms"
]

display(airbnbDF.select(cols))

# COMMAND ----------

from pyspark.sql.functions import col

v = ",\n".join(map(lambda row: "[{}, {}, {}]".format(row[0], row[1], row[2]), airbnbDF.select(col("latitude"),col("longitude"),col("price")/600).collect()))
displayHTML("""
<html>
<head>
 <link rel="stylesheet" href="https://unpkg.com/leaflet@1.3.1/dist/leaflet.css"
   integrity="sha512-Rksm5RenBEKSKFjgI3a41vrjkw4EVPlJ3+OiI65vTjIdo9brlAacEuKOiQ5OFh7cOI1bkDwLqdLw3Zg0cRJAAQ=="
   crossorigin=""/>
 <script src="https://unpkg.com/leaflet@1.3.1/dist/leaflet.js"
   integrity="sha512-/Nsx9X4HebavoBvEBuyp3I7od5tA0UzAxs+j83KgC8PU0kgB4XiK4Lfe4y4cgBtaRJQEIFCW+oC506aPT2L1zw=="
   crossorigin=""></script>
 <script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet.heat/0.2.0/leaflet-heat.js"></script>
</head>
<body>
    <div id="mapid" style="width:700px; height:500px"></div>
  <script>
  var mymap = L.map('mapid').setView([37.7587,-122.4486], 12);
  var tiles = L.tileLayer('http://{s}.tile.osm.org/{z}/{x}/{y}.png', {
    attribution: '&copy; <a href="http://osm.org/copyright">OpenStreetMap</a> contributors',
}).addTo(mymap);
  var heat = L.heatLayer([""" + v + """], {radius: 25}).addTo(mymap);
  </script>
  </body>
  </html>
""")

# COMMAND ----------

# MAGIC %md # ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Transformers, Estimators, and Pipelines
# MAGIC
# MAGIC ## In this video:<br>
# MAGIC
# MAGIC * Transform data using a transformer
# MAGIC * Train an estimator
# MAGIC * Create a pipeline and train a pipeline

# COMMAND ----------

# MAGIC %md-sandbox ## Estimators, Transformers, Pipelines
# MAGIC
# MAGIC Spark's machine learning library, `MLlib`, has three main abstractions:<br><br>
# MAGIC
# MAGIC 1. A **transformer** takes a DataFrame as an input and returns a new DataFrame with one or more columns appended to it
# MAGIC   - Transformers implement a `.transform()` method
# MAGIC 2. An **estimator** takes a DataFrame as an input and returns a model
# MAGIC   - Estimators implements a `.fit()` method.
# MAGIC 3. A **pipeline** combines together transformers and estimators
# MAGIC   - Pipelines implement a `.fit()` method
# MAGIC
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> Note that Spark works by appending columns to immutable DataFrames rather than performing operations in place.

# COMMAND ----------

# MAGIC %md ## Transformers
# MAGIC
# MAGIC See the <a href="http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.Binarizer" target="_blank">Binarizer Docs</a> for more details.

# COMMAND ----------

from pyspark.ml.feature import Binarizer

binarizer = Binarizer(threshold=80, inputCol="review_scores_rating", outputCol="binarized_review_scores_rating")

display(binarizer.transform(airbnbDF))

# COMMAND ----------

# MAGIC %md ## Estimators
# MAGIC
# MAGIC See the <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.StringIndexer.html" target="_blank">StringIndexer Docs</a> and <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.OneHotEncoder.html" target="_blank">OneHotEncoderEstimator Docs</a> for more details.

# COMMAND ----------

from pyspark.ml.feature import StringIndexer

indexer = StringIndexer(inputCol="room_type", outputCol="room_type_index")

# COMMAND ----------

uniqueTypesDF = airbnbDF.select("room_type").distinct() # Use distinct values to demonstrate how StringIndexer works

# COMMAND ----------

indexerModel = indexer.fit(uniqueTypesDF)
indexedDF = indexerModel.transform(uniqueTypesDF)

display(indexedDF)

# COMMAND ----------

from pyspark.ml.feature import OneHotEncoder

encoder = OneHotEncoder(inputCols=["room_type_index"], outputCols=["encoded_room_type"])

encoderModel = encoder.fit(indexedDF)
encodedDF = encoderModel.transform(indexedDF)

display(encodedDF)

# COMMAND ----------

# MAGIC %md ## Pipelines
# MAGIC
# MAGIC See the <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.Pipeline.html" target="_blank">Pipeline Docs</a> for more details.

# COMMAND ----------

from pyspark.ml import Pipeline

pipeline = Pipeline(stages=[
  binarizer,
  indexer, 
  encoder
])

# COMMAND ----------

pipelineModel = pipeline.fit(airbnbDF)
transformedDF = pipelineModel.transform(airbnbDF)

display(transformedDF)

# COMMAND ----------

# MAGIC %md # ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Featurization
# MAGIC
# MAGIC * Perform a train/test split
# MAGIC * Featurize the AirBnB Dataset
# MAGIC * Combine featurization steps into a pipeline

# COMMAND ----------

# MAGIC %md-sandbox ## Train/Test Split
# MAGIC
# MAGIC <div><img src="https://files.training.databricks.com/images/eLearning/ML-Part-1/train-test-split.png" style="height: 400px; margin: 20px"/></div>

# COMMAND ----------

seed = 42
(testDF, trainDF) = airbnbDF.randomSplit((0.20, 0.80), seed=seed)

print(testDF.count(), trainDF.count())

# COMMAND ----------

# MAGIC %md ## Featurization
# MAGIC
# MAGIC *Featurization is the process of creating this input data for a model, which is only as strong as the data it is fed.**  There are a number of common featurization approaches:<br><br>
# MAGIC
# MAGIC * Encoding categorical variables
# MAGIC * Normalizing
# MAGIC * Creating new features
# MAGIC * Handling missing values
# MAGIC * Binning/discretizing

# COMMAND ----------

from pyspark.ml.feature import StringIndexer

iNeighbourhood = StringIndexer(inputCol="neighbourhood_cleansed", outputCol="cat_neighborhood", handleInvalid="skip")
iRoomType = StringIndexer(inputCol="room_type", outputCol="cat_room_type", handleInvalid="skip")
iZipCode = StringIndexer(inputCol="zipcode", outputCol="cat_zipcode", handleInvalid="skip")
iPropertyType = StringIndexer(inputCol="property_type", outputCol="cat_property_type", handleInvalid="skip")
iBedType= StringIndexer(inputCol="bed_type", outputCol="cat_bed_type", handleInvalid="skip")

# COMMAND ----------

from pyspark.ml.feature import OneHotEncoder

oneHotEnc = OneHotEncoder(
  inputCols=["cat_neighborhood", "cat_room_type", "cat_zipcode", "cat_property_type", "cat_bed_type"],
  outputCols=["vec_neighborhood", "vec_room_type", "vec_zipcode", "vec_property_type", "vec_bed_type"]
)

# COMMAND ----------

featureCols = [
 "host_total_listings_count",
 "accommodates",
 "bathrooms",
 "bedrooms",
 "beds",
 "minimum_nights",
 "number_of_reviews",
 "review_scores_rating",
 "review_scores_accuracy",
 "review_scores_cleanliness",
 "review_scores_checkin",
 "review_scores_communication",
 "review_scores_location",
 "review_scores_value",
 "vec_neighborhood", 
 "vec_room_type", 
 "vec_zipcode", 
 "vec_property_type", 
 "vec_bed_type"
]

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Vector Assembler
# MAGIC https://spark.apache.org/docs/3.1.3/api/python/reference/api/pyspark.ml.feature.VectorAssembler.html

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(inputCols=featureCols, outputCol="features")

# COMMAND ----------

# MAGIC %md ## Pipeline
# MAGIC
# MAGIC See the <a href="http://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=pipeline#pyspark.ml.Pipeline" target="_blank">Pipeline Docs</a> for more details.

# COMMAND ----------

from pyspark.ml import Pipeline

featurizationPipeline = Pipeline(stages=[
  iNeighbourhood, 
  iRoomType, 
  iZipCode, 
  iPropertyType, 
  iBedType, 
  oneHotEnc, 
  assembler
])

# COMMAND ----------

display(featurizationPipeline.fit(airbnbDF).transform(airbnbDF))

# COMMAND ----------

# MAGIC %md # ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Model Training and Interpretation
# MAGIC
# MAGIC * Train a Linear regression model
# MAGIC * Train a Gradient Boosted Trees model
# MAGIC * Tuning Xgboost
# MAGIC * Interpret the results

# COMMAND ----------

# MAGIC %md ## Linear Regression
# MAGIC
# MAGIC See the <a href="http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.regression.LinearRegression" target="_blank">LinearRegression Docs</a> for more details.

# COMMAND ----------

from pyspark.ml.regression import LinearRegression

lr = LinearRegression(labelCol="logPrice", featuresCol="features")

piplineFull = Pipeline(stages=[featurizationPipeline, lr])

lrModel = piplineFull.fit(trainDF)

# COMMAND ----------

lr.explainParams()

# COMMAND ----------

lrSummary = lrModel.stages[-1].summary

# COMMAND ----------

lrSummary.r2

# COMMAND ----------

lrModel.stages[-1].coefficients

# COMMAND ----------

lrSummary.pValues

# COMMAND ----------

# MAGIC %md ## Gradient Boosted Trees
# MAGIC
# MAGIC See the <a href="http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.regression.GBTRegressor" target="_blank">GBTRegressor Docs</a> for more details.

# COMMAND ----------

from pyspark.ml.regression import GBTRegressor

gbt = (GBTRegressor()
  .setLabelCol("logPrice")
  .setFeaturesCol("features")
)

piplineFull = Pipeline(stages=[featurizationPipeline, gbt])

gbtModel = piplineFull.fit(trainDF)

# COMMAND ----------

gbtSummary = gbtModel.stages[-1]
gbtSummary.featureImportances

# COMMAND ----------

# MAGIC %md ## Model Evaluation

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

evaluator = RegressionEvaluator(metricName='r2')
print(evaluator.explainParams())

# COMMAND ----------

evaluator.setLabelCol("logPrice")
evaluator.setPredictionCol("prediction")

# COMMAND ----------

evaluator.evaluate(lrModel.transform(testDF))

# COMMAND ----------

evaluator.evaluate(gbtModel.transform(testDF))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tuning Xgboost

# COMMAND ----------

from xgboost.spark import SparkXGBRegressor

# COMMAND ----------

# The next step is to define the model training stage of the pipeline. 
# The following command defines a XgboostRegressor model that takes an input column "features" by default and learns to predict the labels in the "cnt" column.
# Set `num_workers` to the number of spark tasks you want to concurrently run during training xgboost model.
xgb_regressor = SparkXGBRegressor(num_workers=2, label_col="logPrice", missing=0.0)

# COMMAND ----------

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
 
# Define a grid of hyperparameters to test:
#  - maxDepth: maximum depth of each decision tree 
#  - maxIter: iterations, or the total number of trees 
paramGrid = ParamGridBuilder()\
  .addGrid(xgb_regressor.max_depth, [2, 5])\
  .addGrid(xgb_regressor.n_estimators, [10, 100])\
  .build()
 
# Define an evaluation metric.  The CrossValidator compares the true labels with predicted values for each combination of parameters, and calculates this value to determine the best model.
evaluator = RegressionEvaluator(metricName="rmse",
                                labelCol=xgb_regressor.getLabelCol(),
                                predictionCol=xgb_regressor.getPredictionCol())
 
# Declare the CrossValidator, which performs the model tuning.
cv = CrossValidator(estimator=xgb_regressor, evaluator=evaluator, estimatorParamMaps=paramGrid)

# COMMAND ----------

from pyspark.ml import Pipeline
pipeline = Pipeline(stages=[featurizationPipeline, cv])

# COMMAND ----------

pipelineModel = pipeline.fit(trainDF)

# COMMAND ----------

# MAGIC %md
# MAGIC The transform() method of the pipeline model applies the full pipeline to the input dataset. The pipeline applies the feature processing steps to the dataset and then uses the fitted Xgbosot Regressor model to make predictions. The pipeline returns a DataFrame with a new column predictions.

# COMMAND ----------

predictions = pipelineModel.transform(testDF)

# COMMAND ----------

display(predictions)

# COMMAND ----------

metric_eval = evaluator.evaluate(predictions)
print("RMSE on our test set: %g" % metric_eval)

# COMMAND ----------

display(predictions.select("bedrooms", "prediction"))

# COMMAND ----------

# MAGIC %md # ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Prediction and Production
# MAGIC
# MAGIC * Use a trained model to create predictions
# MAGIC * Save the model and predictions
# MAGIC * Explore ways to put models into production

# COMMAND ----------

# MAGIC %md ## Create Predictions

# COMMAND ----------

predictionsDF = gbtModel.transform(testDF)

# COMMAND ----------

display(predictionsDF)

# COMMAND ----------

# MAGIC %md ## Save Models and Predictions

# COMMAND ----------

predictionsDF.write.mode("overwrite").parquet(userhome+"/predictions.parquet")

# COMMAND ----------

path = userhome+"/model"

try:
  gbtModel.save(path)
except:
  dbutils.fs.rm(path, True)
  gbtModel.save(path)

# COMMAND ----------

# MAGIC %md ## Production Options <br>
# MAGIC
# MAGIC * Save predictions to a database or to put behind a REST API
# MAGIC * Predict on an incoming Spark stream
# MAGIC * Export into another frarmework

# COMMAND ----------

