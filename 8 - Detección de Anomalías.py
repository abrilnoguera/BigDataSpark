# Databricks notebook source
from pyspark.sql import SparkSession, functions as F, types as T
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np

# COMMAND ----------

# MAGIC %md
# MAGIC https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html

# COMMAND ----------

np.random.seed(42)

# COMMAND ----------

# create a dataframe
data = [
    {'feature1': 1., 'feature2': 0., 'feature3': 0.3, 'feature4': 0.01},
    {'feature1': 10., 'feature2': 3., 'feature3': 0.9, 'feature4': 0.1},
    {'feature1': 101., 'feature2': 13., 'feature3': 0.9, 'feature4': 0.91},
    {'feature1': 111., 'feature2': 11., 'feature3': 1.2, 'feature4': 1.91},
]
df = spark.createDataFrame(data)
df.show()

# COMMAND ----------

# instantiate a scaler, an isolation forest classifier and convert the data into the appropriate form
scaler = StandardScaler()

# COMMAND ----------

classifier = IsolationForest(contamination=0.3, random_state=42, n_jobs=-1)
x_train = [list(n.values()) for n in data]

# COMMAND ----------

# fit on the data
x_train = scaler.fit_transform(x_train)
clf = classifier.fit(x_train)

# COMMAND ----------

# MAGIC %md
# MAGIC ####Broadcast variables
# MAGIC En PySpark RDD y DataFrame, las variables de transmisión son variables compartidas de solo lectura que se almacenan en caché y están disponibles en todos los nodos de un clúster para que las tareas puedan acceder a ellas o utilizarlas. En lugar de enviar estos datos junto con cada tarea, PySpark distribuye variables de transmisión a los trabajadores utilizando algoritmos de transmisión eficientes para reducir los costos de comunicación.

# COMMAND ----------

# broadcast the scaler and the classifier objects
# remember: broadcasts work well for relatively small objects
SCL = spark.sparkContext.broadcast(scaler)
CLF = spark.sparkContext.broadcast(clf)

# COMMAND ----------

def predict_using_broadcasts(feature1, feature2, feature3, feature4):
    """
    Scale the feature values and use the model to predict
    :return: 1 if normal, -1 if abnormal 0 if something went wrong
    """
    prediction = 0

    x_test = [[feature1, feature2, feature3, feature4]]
    try:
        x_test = SCL.value.transform(x_test)
        prediction = CLF.value.predict(x_test)[0]
    except ValueError:
        import traceback
        traceback.print_exc()
        print('Cannot predict:', x_test)

    return int(prediction)


# COMMAND ----------

udf_predict_using_broadcasts = F.udf(predict_using_broadcasts, T.IntegerType())

df = df.withColumn(
    'prediction',
    udf_predict_using_broadcasts('feature1', 'feature2', 'feature3', 'feature4')
)

# COMMAND ----------

df.show()

# COMMAND ----------

