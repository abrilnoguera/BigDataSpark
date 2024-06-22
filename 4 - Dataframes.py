# Databricks notebook source
# MAGIC %md
# MAGIC ##DataFrames

# COMMAND ----------

#We already have sc and sqlContext for us here
print (sc)
print (sqlContext)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Getting the data

# COMMAND ----------

# MAGIC %sh curl https://files.grouplens.org/datasets/movielens/ml-20m.zip --output /tmp/movielens.csv.zip
# MAGIC unzip /tmp/movielens.csv.zip

# COMMAND ----------

# MAGIC %sh unzip /tmp/movielens.csv.zip -d /tmp/ml-20m

# COMMAND ----------

# MAGIC %fs
# MAGIC ls file:/tmp/ml-20m/ml-20m/

# COMMAND ----------

dbutils.fs.mv("file:/tmp/ml-20m/ml-20m/movies.csv", "dbfs:/tmp/movies.csv")

# COMMAND ----------

movies = (spark.read.format("csv")
.options(header = True, inferSchema = True)
.load("/tmp/movies.csv")) # Keep the dataframe in memory for faster processing 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Basics

# COMMAND ----------

movies.dtypes

# COMMAND ----------

movies.printSchema()

# COMMAND ----------

movies.show(5, truncate=False)

# COMMAND ----------

display(movies)

# COMMAND ----------

# `first()` returns the first row as a Row while
# `head()` and `take()` return `n` number of Row objects
print (movies.first()) # can't supply a value; never a list
print (movies.head(2)) # can optionally supply a value (default: 1);
                      # with n > 1, a list
print (movies.take(2) )# expects a value; always a list

# COMMAND ----------

movies.count()

# COMMAND ----------

movies.explain()

# COMMAND ----------

display(movies)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Import from databricks datasets

# COMMAND ----------

dbfs_dir = '/databricks-datasets/cs110x/ml-20m/data-001'
display(dbutils.fs.ls(dbfs_dir))

# COMMAND ----------

ratings_filename = dbfs_dir + '/ratings.csv' 

# COMMAND ----------

dbutils.fs.head(ratings_filename)

# COMMAND ----------

from pyspark.sql.types import *

ratings_schema = StructType(
  [StructField('userId', IntegerType()),
   StructField('movieId', IntegerType()),
   StructField('rating',FloatType()),
   StructField('timestamp',IntegerType())]
  )

# COMMAND ----------

ratings_df = sqlContext.read.format('com.databricks.spark.csv') \
            .options(header=True, inferSchema=False) \
            .schema(ratings_schema) \
            .load(ratings_filename)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Resume

# COMMAND ----------

ratings_df.show(5)

# COMMAND ----------

ratings_df.count()

# COMMAND ----------

ratings_df.describe().show() 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Analysis on the movies.csv

# COMMAND ----------

movies.show(5)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Get Year

# COMMAND ----------

#transforming the Dataframes
from pyspark.sql.functions import split, regexp_extract

# Regex Year
movies_with_year_df = movies.select('movieId','title',regexp_extract('title',r'\((\d+)\)',1).alias('year'))

# COMMAND ----------

movies_with_year_df.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Split genres

# COMMAND ----------

from pyspark.sql.functions import split, explode
movies_with_genre_df = movies.withColumn('genres', explode(split("genres", "[|]")))

# COMMAND ----------

# MAGIC %md 
# MAGIC ### DataFrames after Transformation

# COMMAND ----------

movies.show(4,truncate = False)
movies_with_genre_df.show(5,truncate = False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Now we will use the inbuilt functionality of Databricks for some insights 

# COMMAND ----------

display(movies_with_genre_df.groupBy('genres').count()) #people love drama

#Below we have a bar chart here we can choose from a lot of other options

# COMMAND ----------

#from here we can look at the count and find that the maximum number of movies are produced in 2009
movies_with_year_df.groupBy('year').count().orderBy('count',ascending = False).show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Now let's move to Ratings
# MAGIC
# MAGIC We already have the movie_df now we will require ratings Lets create the Dataframe

# COMMAND ----------

#We will cache both the dataframes
ratings_df.cache()
movies.cache()
print("both dataframes are in cache now for easy accessibility")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Global Popularity 
# MAGIC It is good to know the most popular movies,and at times it is very hard to just beat popularity [Xavier Amatriain Lecture](https://www.youtube.com/watch?v=bLhq63ygoU8)
# MAGIC  Movies with highest average ratings here we will put a constraint on the no. of reviews given we will discard the movies where the count of ratings is less than 500.

# COMMAND ----------

from pyspark.sql import functions as F

# From ratingsDF, create a movie_ids_with_avg_ratings_df that combines the two DataFrames
movie_ids_with_avg_ratings_df = ratings_df.groupBy('movieId')\
              .agg(F.count(ratings_df.rating).alias("count"), \
                   F.avg(ratings_df.rating).alias("average")
                   )
print ('movie_ids_with_avg_ratings_df:')
movie_ids_with_avg_ratings_df.show(5, truncate=False)

# COMMAND ----------

ratings_df.groupBy('movieId')\
              .agg(F.count(ratings_df.rating).alias("count"), \
                   F.avg(ratings_df.rating).alias("average")
                   ).explain()

# COMMAND ----------

#this df will have names with movie_id- Make it more understandable
movie_names_with_avg_ratings_df = movie_ids_with_avg_ratings_df.join(\
                                    movies,'movieID')
movie_names_with_avg_ratings_df.show(4,truncate = False)

# COMMAND ----------

#so let us see the global popularity
movies_with_500_ratings_or_more = movie_names_with_avg_ratings_df.filter(movie_names_with_avg_ratings_df['count'] >= 500).orderBy('average',ascending = False)
movies_with_500_ratings_or_more.show(10, truncate = False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Operaciones básicas

# COMMAND ----------

# MAGIC %md
# MAGIC ### Select & Selectexp

# COMMAND ----------

from pyspark.sql.functions import expr, col, column
movies.select(
expr("Title"),
col("Title"), ##mas utilizado
column("Title"))\
.show(5, False)

# COMMAND ----------

ratings_df.selectExpr(
"*", # all original columns
"(rating == 5) as max_rating")\
.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Converting to Spark Types (Literals)

# COMMAND ----------

from pyspark.sql.functions import lit
movies.select(expr("*"), lit(1).alias("One")).show(2, False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Add Columns

# COMMAND ----------

movies.withColumn("numberOne", lit(1)).show(2, False)

# COMMAND ----------

ratings_df.withColumn("worst_rating", expr("rating == 1"))\
.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Rename columns

# COMMAND ----------

movies_with_genre_df = movies_with_genre_df.withColumnRenamed("movieId", "ID")

# COMMAND ----------

movies_with_genre_df.columns

# COMMAND ----------

# MAGIC %md
# MAGIC ### Drop columns

# COMMAND ----------

movies.drop("title").columns

# COMMAND ----------

movies.drop("title", "genres").columns

# COMMAND ----------

# MAGIC %md
# MAGIC ### Filter

# COMMAND ----------

ratings_df.filter(col("rating") < 1).show(5)

# COMMAND ----------

ratings_df.where("rating < 1").show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sort

# COMMAND ----------

ratings_df.sort("rating").show(5)
ratings_df.orderBy("rating", "timestamp").show(5)
ratings_df.orderBy(col("rating"), col("timestamp")).show(5)

# COMMAND ----------

from pyspark.sql.functions import desc, asc
ratings_df.orderBy(expr("rating desc")).show(2)
ratings_df.orderBy(col("rating").desc(), col("timestamp").asc()).show(2)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sample

# COMMAND ----------

# in Python
seed = 5
withReplacement = False
fraction = 0.5
ratings_df.sample(withReplacement, fraction, seed).count()

# COMMAND ----------

dataFrames = ratings_df.randomSplit([0.3, 0.7], seed)
dataFrames[0].count()

# COMMAND ----------

dataFrames[1].count()

# COMMAND ----------

dataFrames[1].show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Repartition

# COMMAND ----------

movies_with_genre_df.select('genres').distinct().collect()

# COMMAND ----------

movies_with_genre_df = movies_with_genre_df.repartition(5, col("genres"))