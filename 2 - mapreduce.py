# Databricks notebook source
# MAGIC %md ##Modulo 2
# MAGIC El poder de Spark que opera en conjuntos de datos en memoria es el hecho de que almacena los datos como listas utilizando conjuntos de datos distribuidos resistentes (RDD) que se distribuyen en particiones entre clústeres. Los RDD son una forma rápida de procesamiento de datos, ya que los datos se operan en paralelo según el paradigma de reducción de mapas. Los RDD se pueden usar cuando las operaciones son de bajo nivel. Los RDD generalmente se usan en datos no estructurados como registros o texto. Para datos estructurados y semiestructurados, Spark tiene una mayor abstracción llamada Dataframes. El manejo de datos a través de marcos de datos es extremadamente rápido, ya que están optimizados con el motor Catalyst Opimizatin y el rendimiento es mucho más rápido que los RDD. Además, los marcos de datos también usan Tungsten, que maneja la administración de memoria y la recolección de basura de manera más efectiva.

# COMMAND ----------

# MAGIC %md ##Examples RDD
# MAGIC https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.RDD.html

# COMMAND ----------

#Make RDD
rdd_data = spark.range(10).rdd
rdd_data.collect()

# COMMAND ----------

#RDD to list
list_data = rdd_data.map(lambda row: row[0])
list_data.collect()

# COMMAND ----------

# RDD to Dataframe
rdd_to_df = rdd_data.toDF()
rdd_to_df.show()

# COMMAND ----------

# Dataframe to list
rdd_to_df.rdd.collect()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Diferencia de Paralelizar

# COMMAND ----------

spark.range(10).rdd.map(lambda row: row[0]).collect()

# COMMAND ----------

spark.range(1e7).rdd.map(lambda row: row[0]).reduce(lambda x, y: x + y) 

# COMMAND ----------

spark.sparkContext.parallelize(range(int(1e7))).reduce(lambda x, y: x + y) 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Frases
# MAGIC

# COMMAND ----------

myCollection = "Spark The Definitive Guide : Big Data Processing Made Simple"\
  .split(" ")
words = spark.sparkContext.parallelize(myCollection, 2)
words.setName("myWords")

# COMMAND ----------

words.collect()

# COMMAND ----------

words.name() # myWords

# COMMAND ----------

words.distinct().count()

# COMMAND ----------

words.filter(lambda word: word.startswith("S")).collect()

# COMMAND ----------

words2 = words.map(lambda word: (word, word[0], word.startswith("S")))
words2.filter(lambda record: record[2]).take(5)

# COMMAND ----------

myCollection = "Hola Mundo, esta es la mejor materia del ITBA"\
  .split(" ")
words = spark.sparkContext.parallelize(myCollection, 3)
words.setName("myWords")

# COMMAND ----------

words.map(lambda word: (len(word))).collect()

# COMMAND ----------

words.map(lambda word: (len(word))).reduce(lambda a,b:a +b)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Leyendo un Libro

# COMMAND ----------

import urllib

url = 'https://raw.githubusercontent.com/kuemit/txt_book/master/examples/alice_in_wonderland.txt'

with urllib.request.urlopen(url) as response:
  gzipcontent = response.read()

with open("/tmp/alice_in_wonderland.txt", 'wb') as f:
  f.write(gzipcontent)
 
dbutils.fs.cp("file:/tmp/alice_in_wonderland.txt",'/tmp/alice_in_wonderland.txt')

# COMMAND ----------

book = sc.textFile("dbfs:/tmp/alice_in_wonderland.txt").flatMap(lambda line: line.split(" ")) 
book_p = sc.parallelize(book.collect())

# COMMAND ----------

lower = book_p.map(lambda line: line.lower())

# COMMAND ----------

lower.take(10)

# COMMAND ----------

wordCounts = lower.map(lambda word: (word, 1)).reduceByKey(lambda a,b:a +b)

# COMMAND ----------

wordCounts.take(5)

# COMMAND ----------

invFreq = wordCounts.map(lambda t: (t[1],t[0]))
print(invFreq.top(50))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install Packages

# COMMAND ----------

_ = %pip install nltk

# COMMAND ----------

import nltk
nltk.download('all')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Remover Stopwords y contar

# COMMAND ----------

from nltk.corpus import stopwords

stopwords_list = stopwords.words('english')
stopw = lower.flatMap(lambda x: x.split()).filter(lambda x: x.lower() not in stopwords_list)

# COMMAND ----------

stopw.take(5)

# COMMAND ----------

wordCounts = stopw.map(lambda word: (word, 1)).reduceByKey(lambda a,b:a +b)
invFreq = wordCounts.map(lambda t: (t[1],t[0]))
print(invFreq.top(50))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Leyendo una colección

# COMMAND ----------

import urllib

## https://www.gutenberg.org/

url = 'https://www.gutenberg.org/files/100/100-0.txt'  # The Complete Works of Shakespeare

with urllib.request.urlopen(url) as response:
  gzipcontent = response.read()

with open("/tmp/shakespeare.txt", 'wb') as f:
  f.write(gzipcontent)
 
dbutils.fs.cp("file:/tmp/shakespeare.txt",'/tmp/shakespeare.txt')

# COMMAND ----------

# MAGIC %fs
# MAGIC ls file:/tmp

# COMMAND ----------

collection = sc.textFile("dbfs:/tmp/shakespeare.txt")
collection_p = sc.parallelize(collection.collect(),8)

# COMMAND ----------

collection_p.takeSample(False, 10, 24)

# COMMAND ----------

collection_p.sample(False, .01).count()

# COMMAND ----------

word_counts = (                             # Using parentheses allows inline comments like this
    collection_p.flatMap(lambda line: line.split())  # Split each line into words, flatten the result
       .map(lambda word: word.lower())      # Make each word lowercase
       .map(lambda word: (word, 1))         # Map each word to a key-value pair
       .reduceByKey(lambda x, y: x + y)     # Reduce pairs of values until just one remains for each key
)

# COMMAND ----------

word_counts.take(5)

# COMMAND ----------

top = word_counts.sortBy(lambda kvp: kvp[1], ascending=False)
top.take(5)

# COMMAND ----------

from sklearn.feature_extraction import text

top = top.filter(lambda kvp: kvp[0] not in text.ENGLISH_STOP_WORDS)
top.take(10)

# COMMAND ----------

from sklearn.feature_extraction import text

word_counts = (                             # Using parentheses allows inline comments like this
    collection_p.flatMap(lambda line: line.split())  # Split each line into words, flatten the result
       .map(lambda word: word.lower())      # Make each word lowercase
       .map(lambda word: (word, 1))         # Map each word to a key-value pair
       .reduceByKey(lambda x, y: x + y)     # Reduce pairs of values until just one remains for each key
      .filter(lambda x: x[0] not in text.ENGLISH_STOP_WORDS)
      .sortBy(lambda x: x[1], ascending=False)
)

word_counts.take(10)

# COMMAND ----------

word_counts.toDF().show()