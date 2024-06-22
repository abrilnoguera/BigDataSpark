# Databricks notebook source
# MAGIC %md
# MAGIC <html>
# MAGIC <center>
# MAGIC   <h1 color="green">Parallel Computing</h1> 
# MAGIC   </center>
# MAGIC </html>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://spark.apache.org/images/spark-logo-trademark.png" width="300" alt="logo"/>
# MAGIC
# MAGIC ### Wide Coverage
# MAGIC
# MAGIC <img src="https://spark.apache.org/images/spark-stack.png" alt="coverage"/>
# MAGIC
# MAGIC ### Runs Everywhere
# MAGIC
# MAGIC <img src="https://spark.apache.org/images/spark-runs-everywhere.png" alt="runs-everywhere"/>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Spark Architecture
# MAGIC ![Spark](https://spark.apache.org/docs/latest/img/cluster-overview.png)
# MAGIC Source: <a href="https://spark.apache.org/docs/latest/cluster-overview.html" target="_blank">Spark Official Doc</a>
# MAGIC

# COMMAND ----------

# DBTITLE 1,RDD(Resilient Distributed Dataset)
# MAGIC %md
# MAGIC
# MAGIC * Immutable.
# MAGIC * Lazy transformations
# MAGIC * RDDs are created by starting with a file in the Hadoop file system (or any other Hadoop-supported file system), or an existing Scala collection in the driver program, and transforming it.
# MAGIC * Can persist RDD in memory but has to be requested for that.
# MAGIC * Automatically recover from node failures.
# MAGIC
# MAGIC ## RDD Workflow
# MAGIC ![Spark](https://d1jnx9ba8s6j9r.cloudfront.net/blog/content/ver.1556540029/uploads/2018/09/Picture1-5-768x266.png)
# MAGIC Source: <a href="https://www.edureka.co/blog/spark-architecture/" target="_blank">Edureka</a>
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **The spark context is the main entry point for the spark program as can be seen from the architecture. So first we create the spark context as shown below.**
# MAGIC
# MAGIC ---
# MAGIC ```python
# MAGIC from pyspark import SparkContext
# MAGIC #local indicates to run in local mode
# MAGIC sc = SparkContext("local", "MySparkApp")
# MAGIC
# MAGIC #or
# MAGIC
# MAGIC from pyspark import SparkContext, SparkConf
# MAGIC conf = SparkConf().setAppName(appName).setMaster(master)
# MAGIC sc = SparkContext(conf=conf)
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **PySpark SparkContext contains the following parameters**
# MAGIC
# MAGIC ```python
# MAGIC class pyspark.SparkContext (
# MAGIC    master = None,
# MAGIC    appName = None, 
# MAGIC    sparkHome = None, 
# MAGIC    pyFiles = None, 
# MAGIC    environment = None, 
# MAGIC    batchSize = 0, 
# MAGIC    serializer = PickleSerializer(), 
# MAGIC    conf = None, 
# MAGIC    gateway = None, 
# MAGIC    jsc = None, 
# MAGIC    profiler_cls = <class 'pyspark.profiler.BasicProfiler'>
# MAGIC )
# MAGIC ```
# MAGIC * **master =** URL to the cluster.
# MAGIC * **appName =**  Name for your application to show on the cluster UI
# MAGIC * **sparkHome =** Spark installation directory 
# MAGIC * **pyFiles =**  Zip or python files to send to cluster and add to PYTHONPATH. 
# MAGIC * **environment =** Worker nodes environment vairables. 
# MAGIC * batchSize = Number of Python objects represented as a single Java object. ** 1 to disable batching, 0 to automatically choose the batch size based on object sizes, or -1 to use an unlimited batch size.** 
# MAGIC * **serializer =** RDD serializer. 
# MAGIC * **conf =** An object of {SparkConf} to set all the Spark properties. 
# MAGIC * **gateway =** Use an existing gateway and JVM, otherwise initializing a new JVM. 
# MAGIC * **jsc =**  JavaSparkContext instance. 
# MAGIC *  **profiler_cls =** A class of custom Profiler used to do profiling (the default is pyspark.profiler.BasicProfiler)
# MAGIC
# MAGIC
# MAGIC **Among all those available parameters, master and appName are the one used most.**
# MAGIC

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC **There are two ways to create the RDD**
# MAGIC * Parallelizing an existing collection in your driver program.
# MAGIC * Or referencing a dataset in an external storage system.

# COMMAND ----------

import numpy as np
data = np.arange(1,1000)
print(data[900:])

# COMMAND ----------

#parallelizing existing  collection
distData = sc.parallelize(data)

#we can also specify the number of partitions to cut the dataset into 
distData_ten = sc.parallelize(data,10)

distData.count()

# COMMAND ----------

#retrieve all result
distData.collect()

# COMMAND ----------

#retrieve the first element from RDD
distData.first()

# COMMAND ----------

#retrieve first n elements from RDD
distData.take(20)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC * PySpark can create distributed datasets from any storage source supported by Hadoop, including your local file system, HDFS, Cassandra, HBase, Amazon S3, etc.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC * **filerdd = = sc.textFile("data.txt")**
# MAGIC
# MAGIC **Also supports directory, zip and wildcards** 
# MAGIC   -  textFile("/my/directory"), textFile("/my/directory/*.txt"), and textFile("/my/directory/*.gz")
# MAGIC   
# MAGIC **PySpark supports several other data fromat:**
# MAGIC * **SparkContext.wholeTextFiles:** read a directory containing multiple small text files. 
# MAGIC * **RDD.saveAsPickleFile:** saving RDD in simple file format consisting of pickled python objects with defalt batch size of 10 for pickle serialization using batching.
# MAGIC * **SequenceFile and Hadoop Input/Output Formats**
# MAGIC
# MAGIC **Writable Support**
# MAGIC
# MAGIC <img src="https://i.ibb.co/YX5JZXx/writable.png" alt="writable-format" />

# COMMAND ----------

# MAGIC %fs ls 

# COMMAND ----------

#save the result
rdd = sc.parallelize(range(1, 400)).map(lambda x: (x, x * x))
rdd.saveAsSequenceFile("/sparktest.txt")


# COMMAND ----------

sorted(sc.sequenceFile("/sparktest.txt/").take(10))

# COMMAND ----------

# MAGIC %fs ls  

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **The other way is to use higher level abstraction DataFrame**
# MAGIC

# COMMAND ----------

#create dataframe from existing rdd
df = spark.createDataFrame(rdd)
df.show(7)
df.write.format("json").save("/FileStore/tables/d1ff1")
##/FileStore/tables/ 

# COMMAND ----------

# MAGIC %md 
# MAGIC <img src="https://indatalabs.com/wp-content/uploads/2017/04/pic-2-742x403.png" alt="rdd-df"/>
# MAGIC
# MAGIC
# MAGIC * **RDD:** Low level for raw data and lacks predefined structure. Need self optimization. 
# MAGIC * **Datasets:** Typed data with ability to use spark optimization and also benefits of Spark SQL’s optimized execution engine.
# MAGIC * **DataFrames:** Share the codebase with the Datasets and have the same basic optimizations. In addition, you have optimized code generation, transparent conversions to column based format and an SQL interface.
# MAGIC
# MAGIC Ref: <a href="https://indatalabs.com/blog/convert-spark-rdd-to-dataframe-dataset">indatalabs.com</a>

# COMMAND ----------

# MAGIC %fs ls  /FileStore/tables/d1ff1

# COMMAND ----------

# MAGIC %md
# MAGIC ## Shared Variables 
# MAGIC * **The second abstraction of spark other than main(first) RDD**
# MAGIC * **Spark runs functions in parallel (Default) and ships copy of variable used in function to each task. -- But not across task. **
# MAGIC * **Provides broadcast variables & accumulators.** 
# MAGIC   * Broadcast variables - can be used to cache value in all memory. Shared data can be accessed inside spark functions.
# MAGIC   * Accumulator - for aggregating. Can be used for sum or counter. 
# MAGIC

# COMMAND ----------

#Broadcast
bcast = sc.broadcast([10,20,30,40])
result = bcast.value
print(result)

# COMMAND ----------

#Accumulator
accvalue = sc.accumulator(50) 
def f(x): 
   global accvalue
   accvalue+=x 
#create rdd
rdd = sc.parallelize([10,20,40,30]) 
#loop through it and apply function f
rdd.foreach(f)
#get the final result
result = accvalue.value 
print ("Accumulator Result %i" % (result))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Getting and Setting Number of Partitions**
# MAGIC * spark.conf.get("spark.sql.shuffle.partitions")
# MAGIC * spark.conf.set("spark.sql.shuffle.partitions", N)

# COMMAND ----------

spark.conf.get("spark.sql.shuffle.partitions")

# COMMAND ----------

spark.conf.set("spark.sql.shuffle.partitions", 6)

# COMMAND ----------

spark.conf.get("spark.sql.shuffle.partitions")

# COMMAND ----------

# MAGIC %md 
# MAGIC #RDD Operations (Transformations & Actions)
# MAGIC More Info -[Transformations](https://spark.apache.org/docs/latest/rdd-programming-guide.html#transformations) | [Actions](https://spark.apache.org/docs/latest/rdd-programming-guide.html#actions)
# MAGIC
# MAGIC * **Transformations:**  Operation on applying to RDD creates new RDD. Examples - map(), filter(), groupByKey, reduceByKey() etc.
# MAGIC
# MAGIC * **Actions:** Instruct Spark to perform computation and send the result back to driver. Examples - reduce(), collect(), count(), first(), take(), foreach(), saveAsTextFile() etc.
# MAGIC

# COMMAND ----------

import numpy as np

def returneven(x):
  if x % 2 == 0:
    return x

rdddata = sc.parallelize(np.arange(1,999999),6)
rdddata = rdddata.map(returneven)
rdddata.take(5)

# COMMAND ----------

rdddata = rdddata.filter(lambda x: x is not None)
rdddata.take(5)

# COMMAND ----------

rddAction = rdddata.reduce(lambda x,y: x + y)
rddAction

# COMMAND ----------

# MAGIC %md
# MAGIC # Persistence
# MAGIC
# MAGIC <img src="https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2017/05/RDD-Persistence-and-Caching-Mechanism-in-Apache-Spark-2.jpg" alt="persistance" width="600px"/>
# MAGIC
# MAGIC **Source:**<a href="https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2017/05/RDD-Persistence-and-Caching-Mechanism-in-Apache-Spark-2.jpg" target="_blank">Persistance Tutorial</a>
# MAGIC
# MAGIC * Persistance is a mechanism of caching data across the memory for faster performance.
# MAGIC * Can be achieved using **cache()** and **persist()** methods.
# MAGIC * cache() is default storage level, uses **MEMORY_ONLY** level.
# MAGIC * persist() can use any of the specified below storage level.
# MAGIC * **unpersist()** - to unpersist.
# MAGIC
# MAGIC <img src="https://i.ibb.co/SKBVbGL/persistance.png" alt="persistance level"/>
# MAGIC
# MAGIC * Available storage levels in Python include MEMORY_ONLY, MEMORY_ONLY_2, MEMORY_AND_DISK, MEMORY_AND_DISK_2, DISK_ONLY, and DISK_ONLY_2.
# MAGIC * _2 means replicate each partition on two cluster nodes.
# MAGIC
# MAGIC ### Spark also automatically persists in case of large processing like reduceByKey.
# MAGIC
# MAGIC **More Info -**[Persistence](https://spark.apache.org/docs/latest/rdd-programming-guide.html#rdd-persistence)

# COMMAND ----------

rddpersist = sc.parallelize(range(1, 400))
rddpersist.persist()
rddpersist.first()

# COMMAND ----------

# MAGIC %md
# MAGIC **DAG**
# MAGIC
# MAGIC <img src="https://i.ibb.co/ZhZNcdg/dag-persist.png" alt="persistancedag"/>

# COMMAND ----------

rddpersist.unpersist()
rddpersist.first()

# COMMAND ----------

# MAGIC %md
# MAGIC **DAG**
# MAGIC
# MAGIC <img src="https://i.ibb.co/NtBKrHd/dag-wpersist.png" alt="persistancedag"/>

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Fault tolerance
# MAGIC
# MAGIC * Spark uses **checkpointing** and **Lineage** for fault tolerance.

# COMMAND ----------

# MAGIC %md
# MAGIC ###Checkpointing
# MAGIC <img src="https://i.ibb.co/SBrVFvp/checkpointig.png" alt="checkpointing"/>

# COMMAND ----------

# MAGIC %pwd

# COMMAND ----------

# MAGIC %mkdir /databricks/driver/checkpoint_s

# COMMAND ----------

import os
print(os.listdir('/databricks/driver/'))

# COMMAND ----------

#set the checkpoint directory
spark.sparkContext.setCheckpointDir("/databricks/driver/checkpoint_s")

# COMMAND ----------

rdd1 = sc.parallelize(range(1, 400))
rdd1.checkpoint()

# COMMAND ----------

rdd2 = rdd1.map(lambda x: x*x).first()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### RDD Lineage
# MAGIC <img src="https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2018/01/rdd-lineage.jpg" alt="lineage"/>
# MAGIC
# MAGIC **Source:**<a href="https://data-flair.training/blogs/rdd-lineage/" target="_blank">RDD lineage in Spark</a>
# MAGIC
# MAGIC You can use **toDebugString** to get the spark lineage.

# COMMAND ----------

rdd1 = sc.parallelize(range(1, 400))

# COMMAND ----------

rdd2 = rdd1.filter(lambda x: (x % 2 == 0))

# COMMAND ----------

rdd3 = rdd2.map(lambda x: (x * x))


# COMMAND ----------

print(rdd3.toDebugString().decode("utf-8"))

# COMMAND ----------

# MAGIC %md 
# MAGIC **Example Dataframe**
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %fs ls /FileStore/tables/

# COMMAND ----------

dataFrame = "/FileStore/tables/Demographic_Statistics_By_Zip_Code.csv"
spark.read.format("csv").option("header","true")\
  .option("inferSchema", "true").load(dataFrame)\
  .createOrReplaceTempView("demographic")

# COMMAND ----------

diamonds = spark.sql("select * from demographic")
display(diamonds.select("*"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Graph
# MAGIC
# MAGIC Example from - https://docs.databricks.com/spark/latest/graph-analysis/graphframes/user-guide-python.html

# COMMAND ----------

from functools import reduce
from pyspark.sql.functions import col, lit, when
from graphframes import *

# COMMAND ----------

vertices = sqlContext.createDataFrame([
  ("a", "Alice", 34),
  ("b", "Bob", 36),
  ("c", "Charlie", 30),
  ("d", "David", 29),
  ("e", "Esther", 32),
  ("f", "Fanny", 36),
  ("g", "Gabby", 60)], ["id", "name", "age"])

edges = sqlContext.createDataFrame([
  ("a", "b", "friend"),
  ("b", "c", "follow"),
  ("c", "b", "follow"),
  ("f", "c", "follow"),
  ("e", "f", "follow"),
  ("e", "d", "friend"),
  ("d", "a", "friend"),
  ("a", "e", "friend")
], ["src", "dst", "relationship"])

g = GraphFrame(vertices, edges)
display(g.vertices)

# COMMAND ----------

display(g.inDegrees)

# COMMAND ----------

youngest = g.vertices.groupBy().min("age")
display(youngest)

# COMMAND ----------

#page rank
results = g.pageRank(resetProbability=0.15, tol=0.01)
display(results.vertices)