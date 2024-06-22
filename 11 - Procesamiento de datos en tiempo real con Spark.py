# Databricks notebook source
# MAGIC %md
# MAGIC # Streaming

# COMMAND ----------

# MAGIC %md
# MAGIC ## Streaming Inference

# COMMAND ----------

# MAGIC %md
# MAGIC ### import data

# COMMAND ----------

# MAGIC %sh curl https://raw.githubusercontent.com/Infi-09/Heart-Attack-Project/main/data/heart.csv --output /tmp/heart.csv

# COMMAND ----------

dbutils.fs.mv("file:/tmp/heart.csv", "dbfs:/tmp/heart.csv")

# COMMAND ----------

df = spark.read.format("csv")\
  .option("header", "true")\
  .option("inferSchema", "true")\
  .load("dbfs:/tmp/heart.csv")

# COMMAND ----------

df.show(5)

# COMMAND ----------

df = df.withColumnRenamed("output","label")
df.display()
df.printSchema()

# COMMAND ----------

testDF, trainDF = df.randomSplit([0.3, 0.7])

# COMMAND ----------

testDF.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pipeline ML

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.classification import LogisticRegression
# Create the logistic regression model
lr = LogisticRegression(maxIter=10, regParam= 0.01)
# We create a one hot encoder.
ohe = OneHotEncoder(inputCols = ['sex', 'cp', 'fbs', 'restecg', 'slp', 'exng', 'caa', 'thall'], outputCols=['sex_ohe', 'cp_ohe', 'fbs_ohe', 'restecg_ohe', 'slp_ohe', 'exng_ohe', 'caa_ohe', 'thall_ohe'])
# Input list for scaling
inputs = ['age','trtbps','chol','thalachh','oldpeak']
# We scale our inputs
assembler1 = VectorAssembler(inputCols=inputs, outputCol="features_scaled1")
scaler = MinMaxScaler(inputCol="features_scaled1", outputCol="features_scaled")
# We create a second assembler for the encoded columns.
assembler2 = VectorAssembler(inputCols=['sex_ohe', 'cp_ohe', 'fbs_ohe', 'restecg_ohe', 'slp_ohe', 'exng_ohe', 'caa_ohe', 'thall_ohe','features_scaled'], outputCol="features")
# Create stages list
myStages = [assembler1, scaler, ohe, assembler2,lr]
# Set up the pipeline
pipeline = Pipeline(stages= myStages)
# We fit the model using the training data.
pModel = pipeline.fit(trainDF)
# We transform the data.
trainingPred = pModel.transform(trainDF)
# # We select the actual label, probability and predictions
trainingPred.select('label','probability','prediction').show()

# COMMAND ----------

trainingPred.select('label','probability','prediction').show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Streaming job

# COMMAND ----------

testData = testDF.repartition(10)
#Remove directory in case we rerun it multiple times.
dbutils.fs.rm("FileStore/tables/HeartTest/",True)
#Create a directory
testData.write.format("CSV").option("header",True).save("FileStore/tables/HeartTest/")

# COMMAND ----------

schema = df.schema
# Source
sourceStream=spark.readStream.format("csv").option("header",True).schema(schema).option("ignoreLeadingWhiteSpace",True).option("mode","dropMalformed").option("maxFilesPerTrigger",1).load("dbfs:/FileStore/tables/HeartTest").withColumnRenamed("output","label")

# COMMAND ----------

from pyspark.sql.functions import current_timestamp
streaming_inference=pModel.transform(sourceStream).select('label', 'probability','prediction', current_timestamp())

display(streaming_inference)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Streaming ML
# MAGIC El ajuste se produce en cada lote de datos, de modo que el modelo se actualiza continuamente para reflejar los datos de la secuencia.

# COMMAND ----------

from pyspark.streaming import StreamingContext
ssc = StreamingContext(sc, 1)

# COMMAND ----------

ssc

# COMMAND ----------

# MAGIC %md
# MAGIC ### Regression

# COMMAND ----------

import sys

from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.regression import StreamingLinearRegressionWithSGD

def parse(lp):
    label = float(lp[lp.find('(') + 1: lp.find(',')])
    vec = Vectors.dense(lp[lp.find('[') + 1: lp.find(']')].split(','))
    return LabeledPoint(label, vec)

trainingData = ssc.textFileStream(sys.argv[1]).map(parse).cache()
testData = ssc.textFileStream(sys.argv[2]).map(parse)

numFeatures = 3
model = StreamingLinearRegressionWithSGD(stepSize=0.2, numIterations=25)
model.setInitialWeights([0.0, 0.0, 0.0])

model.trainOn(trainingData)
print(model.predictOnValues(testData.map(lambda lp: (lp.label, lp.features))))

ssc.start()
ssc.awaitTermination().show()

# COMMAND ----------

model

# COMMAND ----------

ssc.start()
ssc.awaitTermination().show()


# COMMAND ----------

ssc.stop()

# COMMAND ----------

# MAGIC %md
# MAGIC ### kmeans

# COMMAND ----------

# MAGIC %sh curl https://raw.githubusercontent.com/apache/spark/master/data/mllib/kmeans_data.txt --output /tmp/kmeans_data.txt

# COMMAND ----------

dbutils.fs.mv("file:/tmp/kmeans_data.txt", "dbfs:/tmp/kmeans_data.txt")

# COMMAND ----------

# MAGIC %sh curl https://raw.githubusercontent.com/apache/spark/master/data/mllib/streaming_kmeans_data_test.txt --output /tmp/streaming_kmeans_data_test.txt

# COMMAND ----------

dbutils.fs.mv("file:/tmp/streaming_kmeans_data_test.txt", "dbfs:/tmp/streaming_kmeans_data_test.txt")

# COMMAND ----------

from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.clustering import StreamingKMeans

# we make an input stream of vectors for training,
# as well as a stream of vectors for testing
def parse(lp):
    label = float(lp[lp.find('(') + 1: lp.find(')')])
    vec = Vectors.dense(lp[lp.find('[') + 1: lp.find(']')].split(','))

    return LabeledPoint(label, vec)

trainingData = sc.textFile("tmp/kmeans_data.txt")\
    .map(lambda line: Vectors.dense([float(x) for x in line.strip().split(' ')]))

testingData = sc.textFile("tmp/streaming_kmeans_data_test.txt").map(parse)

trainingQueue = [trainingData]
testingQueue = [testingData]

trainingStream = ssc.queueStream(trainingQueue)
testingStream = ssc.queueStream(testingQueue)

# We create a model with random clusters and specify the number of clusters to find
model = StreamingKMeans(k=2, decayFactor=1.0).setRandomCenters(3, 1.0, 0)

# Now register the streams for training and testing and start the job,
# printing the predicted cluster assignments on new data points as they arrive.
model.trainOn(trainingStream)

result = model.predictOnValues(testingStream.map(lambda lp: (lp.label, lp.features)))
result.pprint()

ssc.start()

# COMMAND ----------

ssc.awaitTermination().show()

# COMMAND ----------

ssc.stop(stopSparkContext=True, stopGraceFully=True)