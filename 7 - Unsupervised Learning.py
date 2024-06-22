# Databricks notebook source
# MAGIC %sh curl https://raw.githubusercontent.com/databricks/Spark-The-Definitive-Guide/master/data/retail-data/all/online-retail-dataset.csv --output /tmp/online-retail-dataset.csv

# COMMAND ----------

dbutils.fs.mv("file:/tmp/online-retail-dataset.csv", "dbfs:/tmp/retail/online-retail-dataset.csv")

# COMMAND ----------

df = spark.read.format("csv")\
  .option("header", "true")\
  .option("inferSchema", "true")\
  .load("dbfs:/tmp/retail/*.csv")\
  .coalesce(5)
df.cache()

# COMMAND ----------

df.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## String Indexer
# MAGIC https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.StringIndexer.html

# COMMAND ----------

from pyspark.ml.feature import StringIndexer

stringIndexer = StringIndexer(inputCol="Country", outputCol="Country_indx",
    stringOrderType="frequencyDesc")

# COMMAND ----------

df_sales = stringIndexer.fit(df).transform(df)

# COMMAND ----------

df_sales.show(5)

# COMMAND ----------

from pyspark.sql.functions import count
df_sales.groupBy('Country_indx').agg(count('InvoiceNo')).sort('Country_indx').show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Vector Assembler
# MAGIC https://spark.apache.org/docs/3.1.3/api/python/reference/api/pyspark.ml.feature.VectorAssembler.html

# COMMAND ----------

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

# COMMAND ----------

vecAssembler = VectorAssembler(inputCols=["Quantity", "UnitPrice", "Country_indx"], outputCol="features_")
df_vector = vecAssembler.transform(df_sales)

# COMMAND ----------

df_vector.show(5, False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## StandardScaler

# COMMAND ----------

from pyspark.ml.feature import StandardScaler

scaler = StandardScaler(
    inputCol = 'features_', 
    outputCol = 'features',
    withMean = True,
    withStd = True
).fit(df_vector)

df_scaled = scaler.transform(df_vector)

# COMMAND ----------

df_scaled.show(5, False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## PCA
# MAGIC https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.PCA.html

# COMMAND ----------

from pyspark.ml.feature import PCA as PCA

n_components = 2
pca = PCA(
    k = n_components, 
    inputCol = 'features', 
    outputCol = 'pcaFeatures'
).fit(df_scaled)

df_pca = pca.transform(df_scaled)
print('Explained Variance Ratio', pca.explainedVariance.toArray())

# COMMAND ----------

df_pca.select('pcaFeatures').show(5, False)

# COMMAND ----------

import numpy as np

X_pca = df_pca.rdd.map(lambda row: row.pcaFeatures).collect()
X_pca = np.array(X_pca)
y = df_pca.rdd.map(lambda row: row.Country_indx).collect()
y = np.array(y)

# COMMAND ----------

from matplotlib import pyplot as plt

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = 8, 6
plt.rcParams['font.size'] = 12


def plot_pca(X_pca, y):
    """a scatter plot of the 2-dimensional"""
    markers = 's', 'x', 'o'
    colors = list(plt.rcParams['axes.prop_cycle'])
    target = np.unique(y)
    for idx, (t, m) in enumerate(zip(target, markers)):
        subset = X_pca[y == t]
        plt.scatter(subset[:, 0], subset[:, 1], s = 50,
                    c = colors[idx]['color'], label = t, marker = m)

    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend(loc = 'lower left')
    plt.tight_layout()
    plt.show()

# COMMAND ----------

plot_pca(X_pca, y)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Kmeans
# MAGIC https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.clustering.KMeans.html

# COMMAND ----------

# MAGIC %md
# MAGIC ### silhouette_scores

# COMMAND ----------

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

silhouette_scores=[]
evaluator = ClusteringEvaluator(featuresCol='features', \
metricName='silhouette', distanceMeasure='squaredEuclidean')

for K in range(2,11):

    KMeans_=KMeans(featuresCol='features', k=K)

    KMeans_fit=KMeans_.fit(df_scaled)

    KMeans_transform=KMeans_fit.transform(df_scaled) 

    evaluation_score=evaluator.evaluate(KMeans_transform)

    silhouette_scores.append(evaluation_score)

# COMMAND ----------

silhouette_scores

# COMMAND ----------

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,1, figsize =(10,8))
ax.plot(range(2,11),silhouette_scores)
ax.set_xlabel('Number of Clusters')
ax.set_ylabel('Silhouette Score')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model

# COMMAND ----------

kmeans = KMeans(k=6, seed=1)
model = kmeans.fit(df_scaled.select('features'))
transformed = model.transform(df_scaled.select('features'))

# COMMAND ----------

transformed.show(5, False)

# COMMAND ----------

model.clusterCenters()

# COMMAND ----------

summary = model.summary

# COMMAND ----------

summary.clusterSizes

# COMMAND ----------

from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import udf, col

transformed = (transformed.withColumn("xs", vector_to_array("features")))\
    .select(["prediction"] + [col("xs")[i] for i in range(3)])
clustered_data_pd = transformed.toPandas()
clustered_data_pd.head()

# COMMAND ----------

from matplotlib import pyplot as plt
# Visualizing the results
plt.scatter(clustered_data_pd["xs[0]"], clustered_data_pd["xs[2]"], c=clustered_data_pd["prediction"], cmap='viridis')
plt.title("K-means Clustering with PySpark MLlib")
plt.colorbar().set_label("Cluster")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Topic Modeling 

# COMMAND ----------

originalCorpus = sqlContext.read.parquet("/databricks-datasets/news20.binary/data-001/training")
originalCorpus.cache().count()
originalCorpus = originalCorpus.withColumnRenamed("topic","topic_")

# COMMAND ----------

originalCorpus.count()

# COMMAND ----------

display(originalCorpus)

# COMMAND ----------

originalCorpus.show(5)

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import (RegexTokenizer,
                                StopWordsRemover,
                                CountVectorizer,
                                IDF)
from pyspark.ml.clustering import LDA

# COMMAND ----------

# MAGIC %md
# MAGIC ### CountVectorizer
# MAGIC https://spark.apache.org/docs/3.1.3/api/python/reference/api/pyspark.ml.feature.RegexTokenizer.html
# MAGIC https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.CountVectorizer.html
# MAGIC https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.mllib.feature.IDF.html

# COMMAND ----------

tokenizer = RegexTokenizer(gaps=True, pattern="\\W", inputCol="text", outputCol="words", minTokenLength=4)
remover = StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol="tokens", stopWords=['from', 'subject'])
countTF = CountVectorizer(minTF=2, minDF=0.05, maxDF=0.7, vocabSize=2048, inputCol=remover.getOutputCol(), outputCol="tf")
idf = IDF(minDocFreq=20, inputCol=countTF.getOutputCol(), outputCol="idf")

pipeline = Pipeline(stages=[tokenizer, remover, countTF, idf])

# Fit and transform raw contents to features
model = pipeline.fit(originalCorpus)
df_features = model.transform(originalCorpus)
# get selected_words
selected_words = model.stages[-2].vocabulary

# COMMAND ----------

df_features.select('id', 'words', 'tokens', 'text', 'tf', 'idf').show(5)

# COMMAND ----------

df_features.select('idf').show(5, False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Topic Modeling - Latent Dirichlet Allocation

# COMMAND ----------

import time 

k = 20
seed = 24

# Train
t0 = time.time()
lda = LDA().setK(k).setSeed(seed).setFeaturesCol("tf")
model = lda.fit(df_features)
print('It took {:.2f}s to train a model'.format(time.time() - t0))

# COMMAND ----------

from pyspark.sql.functions import (UserDefinedFunction,
                                   explode,
                                   weekofyear,
                                   monotonically_increasing_id,
                                   desc)
from pyspark.sql.types import (StringType,
                               IntegerType,
                               TimestampType,
                               ArrayType,
                               FloatType)
                                                                 
def get_topic_key_words(model, vocab):
    """
    input: 
        model, fitted LDA model
        vocab, list of words in vocab
    output: topic dataframe with key words
    """
    # define get_words function
    get_words = UserDefinedFunction(lambda x: [vocab[i] for i in x], ArrayType(StringType()))
    return model.describeTopics().select('*', get_words('termIndices').alias("keyWords"))

# COMMAND ----------

import pandas as pd 

pd.set_option('max_colwidth', 120)
df_topic = get_topic_key_words(model, selected_words)
df_topic.select('topic', 'keyWords').toPandas()

# COMMAND ----------

def get_topic(df_prediction):
    """
    input: prediction dataframe
    output: prediction dataframe with topic column
    """
    # define get_words function
    get_topic = UserDefinedFunction(lambda x: int(x.values.argsort()[::-1][0]), IntegerType())
    return df_prediction.select('*', get_topic('topicDistribution').alias("topic"))

# COMMAND ----------

# make prediction
df_pred = get_topic(model.transform(df_features)).cache()
df_pred.select('id', 'text', 'topic').show(3)

# COMMAND ----------

# Check group size for each cluster
df_pred.groupby('topic').count().sort(desc("count")) \
    .join(df_topic, 'topic') \
    .select('topic', 'count', 'keyWords') \
    .sort('count', ascending=False)\
    .show(20, False)

# COMMAND ----------

spark.conf.set("spark.sql.execution.arrow.enabled", "false")

# COMMAND ----------

import numpy as np
# let's sample 20% of data for visualization
df_sample = df_pred.select('id', 'idf', 'topic').sample(False, 0.2, seed=99).toPandas()
# let's convert sparse idf features back to dense array for PCA / T-SNE
mat_tf_idf = np.vstack(df_sample.idf.map(lambda v: v.toArray()).tolist())

# COMMAND ----------

df_sample.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### TSNE

# COMMAND ----------

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns

X_tsne_embedded = TSNE(n_components=2, random_state=99).fit_transform(mat_tf_idf)
# convert to dataframe for visualization
df_tsne = pd.DataFrame(X_tsne_embedded, columns=['x_comp', 'y_comp'])
df_tsne['topic'] = df_sample['topic'].copy().values
df_tsne['keyWords'] = df_sample['topic'].map(
    df_topic.select('topic', 'keyWords').toPandas().set_index('topic').to_dict()['keyWords']
).apply(lambda x: ', '.join(x))

# plot
fig, ax = plt.subplots(figsize=(16, 12))
ax.set_title('LDA Topic Modeling Visualization in TSNE 2D Projection')
sns.scatterplot(
    x="x_comp",
    y="y_comp",
    hue='keyWords',
    data=df_tsne,
    ax=ax
)
# Shrink current axis's height by 10% on the bottom
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

# Put a legend below current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=2)