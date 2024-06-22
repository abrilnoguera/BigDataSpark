# Databricks notebook source
# MAGIC %md ##Introduction
# MAGIC The power of Spark which operates on in-memory datasets is the fact that it stores the data as lists using Resilient Distributed Datasets (RDDs) which are themselves distributed in partitions across clusters. RDDs are a fast way of data processing as the data is operated on parallel based on the map-reduce paradigm. RDDs can be be used when the operations are low level. RDDs are typically used on unstructured data like logs or text. For structured and sem-structured data Spark has a higher abstraction called Dataframes.  Handling data through dataframes are extremely fast as they are Optimized using the Catalyst Opimizatin engine and the performance is orders of maganitude faster than RDDs. In addition Dataframes also use Tungsten which handle memory management and garbage collection more effectively

# COMMAND ----------

import urllib

url = 'https://gist.githubusercontent.com/jsdario/1daee22f3f13fe6bc6a343f829565759/raw/3511dc6de6a7bf064c168b4f20b85a20d8f83b91/funes_el_memorioso.txt'

with urllib.request.urlopen(url) as response:
  gzipcontent = response.read()

with open("/tmp/funes_el_memorioso.txt", 'wb') as f:
  f.write(gzipcontent)
 
dbutils.fs.cp("file:/tmp/funes_el_memorioso.txt",'/tmp/funes_el_memorioso.txt')

book = sc.textFile("dbfs:/tmp/funes_el_memorioso.txt").flatMap(lambda line: line.split(" ")) 
words = sc.parallelize(book.collect())

# COMMAND ----------

book.take(5)

# COMMAND ----------

# MAGIC %md #1. RDD

# COMMAND ----------

words.distinct().count()

# COMMAND ----------

words = words.map(lambda word: (word.lower()))

# COMMAND ----------

words.take(5)

# COMMAND ----------

words2 = words.map(lambda word: (word, word[0], word.startswith("f")))

# COMMAND ----------

words2.filter(lambda record: record[0]).take(3)

# COMMAND ----------

words_ = words.map(lambda word: list(word))

# COMMAND ----------

words.take(5)

# COMMAND ----------

words_.take(5)

# COMMAND ----------

words__ = words.flatMap(lambda word: list(word))
words__.take(10)

# COMMAND ----------

words_.count()

# COMMAND ----------

caracteres = words.flatMap(lambda word: list(word))
caracteres.distinct().count()

# COMMAND ----------

words.sortBy(lambda word: len(word) * -1).take(5)

# COMMAND ----------

#https://spark.apache.org/docs/3.1.3/api/python/reference/api/pyspark.sql.DataFrame.randomSplit.html
splits = words.randomSplit([0.5, 0.5], 24)
words.count()

# COMMAND ----------

print("El primer split tiene:", splits[0].count(), " algunos ejemplos:", splits[0].take(5))
print("El segundo split tiene:", splits[1].count(), " algunos ejemplos:", splits[1].take(5))

# COMMAND ----------

words.takeSample(True, 5, 24)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Aplicar funciones
# MAGIC

# COMMAND ----------

def wordLengthReducer(leftWord, rightWord):
  if len(leftWord) > len(rightWord):
    return leftWord
  else:
    return rightWord

words.reduce(wordLengthReducer)

# COMMAND ----------

def lower_clean_str(x):
  punc='!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
  lowercased_str = x.lower()
  for ch in punc:
    lowercased_str = lowercased_str.replace(ch, '')
  return lowercased_str
words_clean = words.map(lower_clean_str)

# COMMAND ----------

words_clean.takeSample(True, 5, 24)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Key Value

# COMMAND ----------

words_clean.map(lambda word: (word.lower(), 1)).take(10)

# COMMAND ----------

keyword = words_clean.keyBy(lambda word: word.lower()[0])
keyword.take(5)

# COMMAND ----------

keyword.keys().take(5)
#keyword.values().take(5)

# COMMAND ----------

keyword.values().take(5)

# COMMAND ----------

keyword.mapValues(lambda word: word.upper()).take(5)

# COMMAND ----------

keyword_ = keyword.flatMapValues(lambda word: word.upper())
keyword_.take(10)

# COMMAND ----------

import random
distinctChars = words.flatMap(lambda word: list(word.lower())).distinct()\
  .collect()
sampleMap = dict(map(lambda c: (c, random.random()), distinctChars))
words.map(lambda word: (word.lower()[0], word))\
  .sampleByKey(True, sampleMap, 2).take(5)

# COMMAND ----------

chars = words_clean.flatMap(lambda word: word.lower())
KVcharacters = chars.map(lambda letter: (letter, 1))
def maxFunc(left, right):
  return max(left, right)
def addFunc(left, right):
  return left + right

# COMMAND ----------

#forma rebuscada
from functools import reduce
KVcharacters.groupByKey().map(lambda row: (row[0], reduce(addFunc, row[1])))\
  .sortBy(lambda x: x[1], ascending=False).take(10)

# COMMAND ----------

#forma mas simple
KVcharacters.reduceByKey(addFunc).sortBy(lambda x: x[1], ascending=False).take(5)

# COMMAND ----------

KVcharacters.aggregateByKey(0, addFunc, addFunc, 2).sortBy(lambda x: x[1], ascending=False).take(5)

# COMMAND ----------

KVcharacters.foldByKey(0, addFunc).sortBy(lambda x: x[1], ascending=False).take(5)

# COMMAND ----------

distinctChars = words.flatMap(lambda word: list(word.lower())).distinct()
keyedChars = distinctChars.map(lambda c: (c, random.random()))
outputPartitions = 4
#KVcharacters.join(keyedChars).count()
KVcharacters.join(keyedChars, outputPartitions).take(5)

# COMMAND ----------

# MAGIC %md ##1a. RDD - Select all columns of tables

# COMMAND ----------

from pyspark import SparkContext 
rdd = sc.textFile( "/FileStore/tables/tendulkar.csv")
rdd.map(lambda line: (line.split(","))).take(5)

# COMMAND ----------

# MAGIC %md ## 1b.RDD - Select columns 1 to 4

# COMMAND ----------

from pyspark import SparkContext 
rdd = sc.textFile( "/FileStore/tables/tendulkar.csv")
rdd.map(lambda line: (line.split(",")[0:4])).take(5)

# COMMAND ----------

# MAGIC %md ##1c. RDD - Select specific columns 0, 10

# COMMAND ----------

from pyspark import SparkContext 
rdd = sc.textFile( "/FileStore/tables/tendulkar.csv")
df=rdd.map(lambda line: (line.split(",")))
df.map(lambda x: (x[10],x[0])).take(5)

# COMMAND ----------

# MAGIC %md ##1d. RDD -  Filter rows on specific condition

# COMMAND ----------

from pyspark import SparkContext 
rdd = sc.textFile( "/FileStore/tables/tendulkar.csv")
df=(rdd.map(lambda line: line.split(",")[:])
      .filter(lambda x: x !="DNB")
      .filter(lambda x: x!= "TDNB")
      .filter(lambda x: x!="absent")
      .map(lambda x: [x[0].replace("*","")] + x[1:]))

df.take(5)


# COMMAND ----------

# MAGIC %md ##1e. RDD - Find rows where Runs > 50

# COMMAND ----------

from pyspark import SparkContext 
rdd = sc.textFile( "/FileStore/tables/tendulkar.csv")
df=rdd.map(lambda line: (line.split(",")))
df=rdd.map(lambda line: line.split(",")[0:4]) \
   .filter(lambda x: x[0] not in ["DNB", "TDNB", "absent"])
df1=df.map(lambda x: [x[0].replace("*","")] + x[1:4])
header=df1.first()
df2=df1.filter(lambda x: x !=header)
df3=df2.map(lambda x: [float(x[0])] +x[1:4])
df3.filter(lambda x: x[0]>=50).take(10)





# COMMAND ----------

# MAGIC %md ##1f. RDD - groupByKey() and reduceByKey()

# COMMAND ----------

from pyspark import SparkContext 
from pyspark.mllib.stat import Statistics
rdd = sc.textFile( "/FileStore/tables/tendulkar.csv")
df=rdd.map(lambda line: (line.split(",")))
df=rdd.map(lambda line: line.split(",")[0:]) \
   .filter(lambda x: x[0] not in ["DNB", "TDNB", "absent"])
df1=df.map(lambda x: [x[0].replace("*","")] + x[1:])
header=df1.first()
df2=df1.filter(lambda x: x !=header)
df3=df2.map(lambda x: [float(x[0])] +x[1:])
df4 = df3.map(lambda x: (x[10],x[0]))
df5=df4.reduceByKey(lambda a,b: a+b,1)


# COMMAND ----------

df5.take(3)