# Databricks notebook source
# MAGIC %md
# MAGIC # Binary Classification Pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC The Spark MLlib Pipelines API provides higher-level API built on top of DataFrames for constructing ML pipelines.
# MAGIC You can read more about the Pipelines API in the [MLlib programming guide](https://spark.apache.org/docs/latest/ml-guide.html).
# MAGIC
# MAGIC **Binary Classification** is the task of predicting a binary label.
# MAGIC For example, is an email spam or not spam? Should I show this ad to this user or not? Will it rain tomorrow or not?
# MAGIC This notebook illustrates algorithms for making these types of predictions.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dataset Review

# COMMAND ----------

# MAGIC %md
# MAGIC The Adult dataset is publicly available at the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Adult).
# MAGIC This data derives from census data and consists of information about 48842 individuals and their annual income.
# MAGIC You can use this information to predict if an individual earns **<=50K or >50k** a year.
# MAGIC The dataset consists of both numeric and categorical variables.
# MAGIC
# MAGIC Attribute Information:
# MAGIC
# MAGIC - age: continuous
# MAGIC - workclass: Private,Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked
# MAGIC - fnlwgt: continuous
# MAGIC - education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc...
# MAGIC - education-num: continuous
# MAGIC - marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent...
# MAGIC - occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners...
# MAGIC - relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried
# MAGIC - race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black
# MAGIC - sex: Female, Male
# MAGIC - capital-gain: continuous
# MAGIC - capital-loss: continuous
# MAGIC - hours-per-week: continuous
# MAGIC - native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany...
# MAGIC
# MAGIC Target/Label: - <=50K, >50K
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data

# COMMAND ----------

# MAGIC %md
# MAGIC The Adult dataset is available in databricks-datasets. Read in the data using the CSV data source for Spark and rename the columns appropriately.

# COMMAND ----------

# MAGIC %fs ls databricks-datasets/adult/adult.data

# COMMAND ----------

# MAGIC %fs head databricks-datasets/adult/adult.data

# COMMAND ----------

from pyspark.sql.types import DoubleType, StringType, StructField, StructType

schema = StructType([
  StructField("age", DoubleType(), False),
  StructField("workclass", StringType(), False),
  StructField("fnlwgt", DoubleType(), False),
  StructField("education", StringType(), False),
  StructField("education_num", DoubleType(), False),
  StructField("marital_status", StringType(), False),
  StructField("occupation", StringType(), False),
  StructField("relationship", StringType(), False),
  StructField("race", StringType(), False),
  StructField("sex", StringType(), False),
  StructField("capital_gain", DoubleType(), False),
  StructField("capital_loss", DoubleType(), False),
  StructField("hours_per_week", DoubleType(), False),
  StructField("native_country", StringType(), False),
  StructField("income", StringType(), False)
])

dataset = spark.read.format("csv").schema(schema).load("/databricks-datasets/adult/adult.data")
cols = dataset.columns

# COMMAND ----------

display(dataset)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preprocess Data
# MAGIC
# MAGIC To use algorithms like Logistic Regression, you must first convert the categorical variables in the dataset into numeric variables.
# MAGIC There are two ways to do this.
# MAGIC
# MAGIC * Category Indexing
# MAGIC
# MAGIC   This is basically assigning a numeric value to each category from {0, 1, 2, ...numCategories-1}.
# MAGIC   This introduces an implicit ordering among your categories, and is more suitable for ordinal variables (eg: Poor: 0, Average: 1, Good: 2)
# MAGIC
# MAGIC * One-Hot Encoding
# MAGIC
# MAGIC   This converts categories into binary vectors with at most one nonzero value (eg: (Blue: [1, 0]), (Green: [0, 1]), (Red: [0, 0]))
# MAGIC
# MAGIC This notebook uses a combination of [StringIndexer] and, depending on your Spark version, either [OneHotEncoder] or [OneHotEncoderEstimator] to convert the categorical variables.
# MAGIC `OneHotEncoder` and `OneHotEncoderEstimator` return a [SparseVector]. 
# MAGIC
# MAGIC Since there is more than one stage of feature transformations, use a [Pipeline] to tie the stages together.
# MAGIC This simplifies the code.
# MAGIC
# MAGIC [StringIndexer]: http://spark.apache.org/docs/latest/ml-features.html#stringindexer
# MAGIC [OneHotEncoderEstimator]: https://spark.apache.org/docs/2.4.5/api/python/pyspark.ml.html?highlight=one%20hot%20encoder#pyspark.ml.feature.OneHotEncoderEstimator
# MAGIC [SparseVector]: https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.linalg.SparseVector.html#pyspark.ml.linalg.SparseVector
# MAGIC [Pipeline]: https://spark.apache.org/docs/latest/ml-pipeline.html#ml-pipelines
# MAGIC [OneHotEncoder]: https://spark.apache.org/docs/latest/ml-features.html#onehotencoder

# COMMAND ----------

import pyspark
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler

from distutils.version import LooseVersion

categoricalColumns = ["workclass", "education", "marital_status", "occupation", "relationship", "race", "sex", "native_country"]
stages = [] # stages in Pipeline
for categoricalCol in categoricalColumns:
    # Category Indexing with StringIndexer
    stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol + "Index")
    # Use OneHotEncoder to convert categorical variables into binary SparseVectors
    if LooseVersion(pyspark.__version__) < LooseVersion("3.0"):
        from pyspark.ml.feature import OneHotEncoderEstimator
        encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
    else:
        from pyspark.ml.feature import OneHotEncoder
        encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
    # Add stages.  These are not run here, but will run all at once later on.
    stages += [stringIndexer, encoder]

# COMMAND ----------

# MAGIC %md
# MAGIC The above code basically indexes each categorical column using the `StringIndexer`,
# MAGIC and then converts the indexed categories into one-hot encoded variables.
# MAGIC The resulting output has the binary vectors appended to the end of each row.
# MAGIC
# MAGIC Use the `StringIndexer` again to encode labels to label indices.

# COMMAND ----------

# Convert label into label indices using the StringIndexer
label_stringIdx = StringIndexer(inputCol="income", outputCol="label")
stages += [label_stringIdx]

# COMMAND ----------

# MAGIC %md
# MAGIC Use a `VectorAssembler` to combine all the feature columns into a single vector column.
# MAGIC This includes both the numeric columns and the one-hot encoded binary vector columns in the dataset.

# COMMAND ----------

# Transform all features into a vector using VectorAssembler
numericCols = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]

# COMMAND ----------

partialPipeline = Pipeline().setStages(stages)
partialPipeline.getStages()

# COMMAND ----------

# MAGIC %md
# MAGIC Run the stages as a Pipeline. This puts the data through all of the feature transformations in a single call.

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression
  
#partialPipeline = Pipeline().setStages(stages)
pipelineModel = partialPipeline.fit(dataset)
preppedDataDF = pipelineModel.transform(dataset)

# COMMAND ----------

display(preppedDataDF)

# COMMAND ----------

# Fit model to prepped data
lrModel = LogisticRegression().fit(preppedDataDF)

# ROC for training data
display(lrModel, preppedDataDF, "ROC")

# COMMAND ----------

display(lrModel, preppedDataDF)

# COMMAND ----------

# Keep relevant columns
selectedcols = ["label", "features"] + cols
dataset = preppedDataDF.select(selectedcols)
display(dataset)

# COMMAND ----------

### Randomly split data into training and test sets. set seed for reproducibility
(trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed=100)
print(trainingData.count())
print(testData.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fit and Evaluate Models
# MAGIC
# MAGIC Now, try out some of the Binary Classification algorithms available in the Pipelines API.
# MAGIC
# MAGIC Out of these algorithms, the below are also capable of supporting multiclass classification with the Python API:
# MAGIC - Decision Tree Classifier
# MAGIC - Random Forest Classifier
# MAGIC
# MAGIC These are the general steps to build the models:
# MAGIC - Create initial model using the training set
# MAGIC - Tune parameters with a `ParamGrid` and 5-fold Cross Validation
# MAGIC - Evaluate the best model obtained from the Cross Validation using the test set
# MAGIC
# MAGIC Use the `BinaryClassificationEvaluator` to evaluate the models, which uses [areaUnderROC] as the default metric.
# MAGIC
# MAGIC [areaUnderROC]: https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Logistic Regression
# MAGIC
# MAGIC You can read more about [Logistic Regression] from the [classification and regression] section of MLlib Programming Guide.
# MAGIC In the Pipelines API, you can now perform Elastic-Net Regularization with Logistic Regression, as well as other linear methods.
# MAGIC
# MAGIC [classification and regression]: https://spark.apache.org/docs/latest/ml-classification-regression.html
# MAGIC [Logistic Regression]: https://spark.apache.org/docs/latest/ml-classification-regression.html#logistic-regression

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression

# Create initial LogisticRegression model
lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10)

# Train model with Training Data
lrModel = lr.fit(trainingData)

# COMMAND ----------

# ROC for training data
display(lrModel, dataset, "ROC")

# COMMAND ----------

# Make predictions on test data using the transform() method.
# LogisticRegression.transform() will only use the 'features' column.
predictions = lrModel.transform(testData)

# COMMAND ----------

# View model's predictions and probabilities of each prediction class
# You can select any columns in the above schema to view as well
selected = predictions.select("label", "prediction", "probability", "age", "occupation")
display(selected)

# COMMAND ----------

# MAGIC %md
# MAGIC Use `BinaryClassificationEvaluator` to evaluate the model. 

# COMMAND ----------

# MAGIC %md
# MAGIC Note that the default metric for the ``BinaryClassificationEvaluator`` is ``areaUnderROC``
# MAGIC

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Evaluate model
evaluator = BinaryClassificationEvaluator()
evaluator.evaluate(predictions)

# COMMAND ----------

evaluator.getMetricName()

# COMMAND ----------

# Evaluate model
evaluator = BinaryClassificationEvaluator(metricName='areaUnderPR')#metricName='areaUnderROC'
evaluator.evaluate(predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC The evaluator accepts two kinds of metrics - areaUnderROC and areaUnderPR.
# MAGIC Set it to areaUnderPR by using evaluator.setMetricName("areaUnderPR").
# MAGIC
# MAGIC Now, tune the model using `ParamGridBuilder` and `CrossValidator`.
# MAGIC
# MAGIC You can use `explainParams()` to print a list of all parameters and their definitions.
# MAGIC

# COMMAND ----------

print(lr.explainParams())

# COMMAND ----------

# MAGIC %md
# MAGIC Using three values for regParam, three values for maxIter, and two values for elasticNetParam,
# MAGIC the grid includes 3 x 3 x 3 = 27 parameter settings for CrossValidator.

# COMMAND ----------

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Create ParamGrid for Cross Validation
paramGrid = (ParamGridBuilder()
             .addGrid(lr.regParam, [0.01, 0.5, 2.0])
             .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
             .addGrid(lr.maxIter, [1, 5, 10])
             .build())

# COMMAND ----------

# Create 5-fold CrossValidator
cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)

# Run cross validations
cvModel = cv.fit(trainingData)
# this will likely take a fair amount of time because of the amount of models that we're creating and testing

# COMMAND ----------

# Use the test set to measure the accuracy of the model on new data
predictions = cvModel.transform(testData)

# COMMAND ----------

# cvModel uses the best model found from the Cross Validation
# Evaluate best model
evaluator.evaluate(predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC You can also access the model's feature weights and intercepts.

# COMMAND ----------

print('Model Intercept: ', cvModel.bestModel.intercept)

# COMMAND ----------

weights = cvModel.bestModel.coefficients
weights = [(float(w),) for w in weights]  # convert numpy type to float, and to tuple
weightsDF = spark.createDataFrame(weights, ["Feature Weight"])
display(weightsDF)

# COMMAND ----------

# View best model's predictions and probabilities of each prediction class
selected = predictions.select("label", "prediction", "probability", "age", "occupation")
display(selected)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Decision Trees
# MAGIC
# MAGIC You can read more about [Decision Trees](http://spark.apache.org/docs/latest/mllib-decision-tree.html) in the Spark MLLib Programming Guide.
# MAGIC The Decision Trees algorithm is popular because it handles categorical
# MAGIC data and works out of the box with multiclass classification tasks.

# COMMAND ----------

from pyspark.ml.classification import DecisionTreeClassifier

# Create initial Decision Tree Model
dt = DecisionTreeClassifier(labelCol="label", featuresCol="features", maxDepth=3)

# Train model with Training Data
dtModel = dt.fit(trainingData)

# COMMAND ----------

# MAGIC %md
# MAGIC You can extract the number of nodes in the decision tree as well as the
# MAGIC tree depth of the model.

# COMMAND ----------

print("numNodes = ", dtModel.numNodes)
print("depth = ", dtModel.depth)

# COMMAND ----------

display(dtModel)

# COMMAND ----------

# Make predictions on test data using the Transformer.transform() method.
predictions = dtModel.transform(testData)

# COMMAND ----------

predictions.printSchema()

# COMMAND ----------

# View model's predictions and probabilities of each prediction class
selected = predictions.select("label", "prediction", "probability", "age", "occupation")
display(selected)

# COMMAND ----------

# MAGIC %md
# MAGIC Evaluate the Decision Tree model with
# MAGIC `BinaryClassificationEvaluator`.

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator()
evaluator.evaluate(predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC Entropy and the Gini coefficient are the supported measures of impurity for Decision Trees. This is ``Gini`` by default. Changing this value is simple, ``model.setImpurity("Entropy")``.
# MAGIC

# COMMAND ----------

dt.getImpurity()

# COMMAND ----------

# MAGIC %md
# MAGIC Now tune the model with using `ParamGridBuilder` and `CrossValidator`.
# MAGIC
# MAGIC With three values for maxDepth and three values for maxBin, the grid has 4 x 3 = 12 parameter settings for `CrossValidator`. 
# MAGIC

# COMMAND ----------

# Create ParamGrid for Cross Validation
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
paramGrid = (ParamGridBuilder()
             .addGrid(dt.maxDepth, [2, 4, 6, 10])
             .addGrid(dt.maxBins, [20, 40, 80])
             .build())

# COMMAND ----------

# Create 4-fold CrossValidator
cv = CrossValidator(estimator=dt, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=4)

# Run cross validations
cvModel = cv.fit(trainingData)
# Takes ~5 minutes

# COMMAND ----------

print("numNodes = ", cvModel.bestModel.numNodes)
print("depth = ", cvModel.bestModel.depth)

# COMMAND ----------

# Use test set to measure the accuracy of the model on new data
predictions = cvModel.transform(testData)

# COMMAND ----------

# cvModel uses the best model found from the Cross Validation
# Evaluate best model
evaluator.evaluate(predictions)

# COMMAND ----------

# View Best model's predictions and probabilities of each prediction class
selected = predictions.select("label", "prediction", "probability", "age", "occupation")
display(selected)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Random Forest
# MAGIC
# MAGIC Random Forests uses an ensemble of trees to improve model accuracy.
# MAGIC You can read more about [Random Forest] from the [classification and regression] section of MLlib Programming Guide.
# MAGIC
# MAGIC [classification and regression]: https://spark.apache.org/docs/latest/ml-classification-regression.html
# MAGIC [Random Forest]: https://spark.apache.org/docs/latest/ml-classification-regression.html#random-forests

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier

# Create an initial RandomForest model.
rf = RandomForestClassifier(labelCol="label", featuresCol="features")

# Train model with Training Data
rfModel = rf.fit(trainingData)

# COMMAND ----------

# Make predictions on test data using the Transformer.transform() method.
predictions = rfModel.transform(testData)

# COMMAND ----------

predictions.printSchema()

# COMMAND ----------

# View model's predictions and probabilities of each prediction class
selected = predictions.select("label", "prediction", "probability", "age", "occupation")
display(selected)

# COMMAND ----------

# MAGIC %md
# MAGIC Evaluate the Random Forest model with `BinaryClassificationEvaluator`.

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Evaluate model
evaluator = BinaryClassificationEvaluator()
evaluator.evaluate(predictions)

# COMMAND ----------

from itertools import chain
import pandas as pd

attrs = sorted(
    (attr["idx"], attr["name"])
    for attr in (
        chain(*trainingData.schema["features"].metadata["ml_attr"]["attrs"].values())
    )
) 

dic_feats = [
    (name, rfModel.featureImportances[idx])
    for idx, name in attrs
    if rfModel.featureImportances[idx]
]

pd.DataFrame(dic_feats, columns=['feature', 'value']).sort_values(by='value')

# COMMAND ----------

# MAGIC %md
# MAGIC Now tune the model with `ParamGridBuilder` and `CrossValidator`.
# MAGIC
# MAGIC With three values for maxDepth, two values for maxBin, and two values for numTrees,
# MAGIC the grid has 3 x 2 x 2 = 12 parameter settings for `CrossValidator`.

# COMMAND ----------

# Create ParamGrid for Cross Validation
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

paramGrid = (ParamGridBuilder()
             .addGrid(rf.maxDepth, [2, 4, 6])
             .addGrid(rf.maxBins, [20, 60])
             .addGrid(rf.numTrees, [5, 20])
             .build())

# COMMAND ----------

# Create 5-fold CrossValidator
cv = CrossValidator(estimator=rf, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)

# Run cross validations.  This can take about 6 minutes since it is training over 20 trees!
cvModel = cv.fit(trainingData)

# COMMAND ----------

# Use the test set to measure the accuracy of the model on new data
predictions = cvModel.transform(testData)

# COMMAND ----------

# cvModel uses the best model found from the Cross Validation
# Evaluate best model
evaluator.evaluate(predictions)

# COMMAND ----------

# View Best model's predictions and probabilities of each prediction class
selected = predictions.select("label", "prediction", "probability", "age", "occupation")
display(selected)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Make Predictions
# MAGIC As Random Forest gives the best areaUnderROC value, use the bestModel obtained from Random Forest for deployment,
# MAGIC and use it to generate predictions on new data.
# MAGIC Instead of new data, this example generates predictions on the entire dataset.

# COMMAND ----------

bestModel = cvModel.bestModel

# COMMAND ----------

bestModel

# COMMAND ----------

# Generate predictions for entire dataset
finalPredictions = bestModel.transform(dataset)

# COMMAND ----------

# Evaluate best model
evaluator.evaluate(finalPredictions)

# COMMAND ----------

# MAGIC %md
# MAGIC This example shows predictions grouped by age and occupation.

# COMMAND ----------

finalPredictions.createOrReplaceTempView("finalPredictions")

# COMMAND ----------

# MAGIC %md
# MAGIC In an operational environment, analysts may use a similar machine learning pipeline to obtain predictions on new data, organize it into a table and use it for analysis or lead targeting.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT occupation, prediction, count(*) AS count
# MAGIC FROM finalPredictions
# MAGIC GROUP BY occupation, prediction
# MAGIC ORDER BY occupation
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT age, prediction, count(*) AS count
# MAGIC FROM finalPredictions
# MAGIC GROUP BY age, prediction
# MAGIC ORDER BY age