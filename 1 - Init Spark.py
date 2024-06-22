# Databricks notebook source
spark

# COMMAND ----------

sc

# COMMAND ----------

df = spark.createDataFrame([['a', 'b','c'],
                            ['a1', 'b1', 'c1']])
df.show()

# COMMAND ----------

# DBTITLE 1,Example Parallelize
#Distribuya una colección local de Python para formar un RDD. 
movies = ['dark knight', 'barbie', 'openheimer', 'dunkirk', 'pulp fiction', 'avatar']
movies_rdd = sc.parallelize(movies, 4)

# COMMAND ----------

#Recuperar todos los elementos del RDD/DataFrame/Dataset (de todos los nodos) al nodo controlador. 
#WARNING! Se utiliza en conjuntos de datos más pequeños, generalmente después de filter(), group(), count(), etc. La recuperación de un conjunto de datos más grande da como resultado falta de memoria.
movies_rdd.collect()

# COMMAND ----------

#Funciona escaneando primero una partición y utiliza los resultados de esa partición para estimar la cantidad de particiones adicionales necesarias para satisfacer el límite.
movies_rdd.take(3)

# COMMAND ----------

# DBTITLE 1,Get Partitions
#se crean con el número total de cores en todos los nodos ejecutores.
movies_rdd.getNumPartitions()

# COMMAND ----------

sc.parallelize(movies).getNumPartitions()

# COMMAND ----------

# DBTITLE 1,Set Partitions
 someRDD = sc.parallelize(range(100000000),10)
 someRDD.getNumPartitions()

# COMMAND ----------

# Se utiliza sólo para reducir el número de particiones. Esta es una versión optimizada o mejorada de repartition() donde el movimiento de los datos entre las particiones es menor mediante la fusión.
someRDD.coalesce(8).getNumPartitions()

# COMMAND ----------

# DBTITLE 1,Group By primer letra del título
#groupBy mezcla todo el conjunto de datos en la red según la clave, mientras que ReduceByKey calculará sumas locales para cada clave en cada partición y combinará esas sumas locales en sumas más grandes después de la mezcla.
movies_rdd.map(lambda word: word.title()). \
    groupBy(lambda title: title[0]). \
    map(lambda group: (group[0], len(group[1]))).collect()

# COMMAND ----------

movies_rdd.groupByKey().collect()

# COMMAND ----------

# DBTITLE 1,Reduced By
#groupByKey() agrupa todos los valores con la misma clave en un nuevo RDD, mientras que reduceByKey() realiza una función de reducción en todos los valores con la misma clave para producir un único valor reducido.
movies_rdd.map(lambda word: word.title()). \
    map(lambda group: (group[0], len(group[1]))). \
    reduceByKey(lambda a,b: a+b).collect()

# COMMAND ----------

displayHTML("<img src ='https://qph.cf2.quoracdn.net/main-qimg-e83bdec61ab6576f6a66fd7252b504e8'>")

# COMMAND ----------

ejemplo = movies_rdd.map(lambda word: word.title()). \
    map(lambda groupa: (group[0], len(group[1]))). \
    reduceByKey(lambda a,b: a+b)

# COMMAND ----------

ejemplo.take(3)

# COMMAND ----------

rcc = movies_rdd.map(lambda word: word.title()). \
    map(lambda group: (group[0], len(group[1]))). \
    reduceByKey(lambda a,b: a+b)
df_movies = rcc.toDF(["Word", 'Count'])

# COMMAND ----------

df_movies.printSchema()

# COMMAND ----------

df_movies.show()

# COMMAND ----------

movies_rdd.map(lambda x: (x,'Movie')).toDF(['Title', 'Categorie']).show()

# COMMAND ----------

#movies_rdd.map(lambda x: (x, )).collect()

# COMMAND ----------

# DBTITLE 1,Armar un dataframe que cuente peliculas según longitud del título
##Pista: Lo pueden resolver con Reduce By Key
movies_rdd.map(lambda word: word.title()). \
    map(lambda group: (len(group),1)). \
    reduceByKey(lambda a,b: a+b).collect()

# COMMAND ----------


movies_size = movies_rdd.map(lambda word: word.title()). \
    map(lambda group: (len(group), 1)). \
    reduceByKey(lambda a,b: a+b)

# COMMAND ----------

# DBTITLE 1,Ordenar por cantidad y obtener la longitud de titulo mas repetida
movies_size.toDF(['size', 'cantidad']).sort('cantidad', ascending=False).show()

# COMMAND ----------

from pyspark.sql.functions import *
movies_size.toDF(['size', 'cantidad']).sort(desc('cantidad'), asc('size')).show()