# Databricks notebook source
# MAGIC %md
# MAGIC <div style="text-align: center;">
# MAGIC     <h1> Trabajo Práctico 2 </h1>
# MAGIC     <h2> Feature Engineering & Machine Learning Model </h2>
# MAGIC     <h3>Abril Noguera - Abril Schafer - Ian Dalton</h3>
# MAGIC </div>

# COMMAND ----------

# Instalación de librerias
!pip install mlflow --quiet

# COMMAND ----------

# Importación de librerias
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import *
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.regression import *
from pyspark.ml.classification import *
from pyspark.ml.evaluation import RegressionEvaluator, BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import mlflow
import mlflow.spark
import mlflow.xgboost
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from sklearn.metrics import mean_squared_error, roc_curve, precision_recall_curve, confusion_matrix, roc_auc_score, precision_score, recall_score, accuracy_score 

# COMMAND ----------

# Lectura del archivo preprocesado
df = spark.read.parquet("/dbfs/FileStore/preprocessed_df.parquet")

# COMMAND ----------

# Muestra de la base
df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Engineering
# MAGIC Las transformaciones se encapsulan en un `Pipeline` para automatizar y asegurar la consistencia del procesamiento.

# COMMAND ----------

# Etapas del pipeline
stages = []

# COMMAND ----------

# MAGIC %md
# MAGIC ### Definición de Variables Predictoras para el Modelo Predictivo

# COMMAND ----------

# Muestra de columnas de la base
print(df.columns)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Variables Eliminadas
# MAGIC Para simplificar el modelo y mejorar su interpretabilidad, se han eliminado las siguientes categorías de variables:
# MAGIC - **Identificadores y URLs:** `id`, `source`, `listing_url`, `scrape_id`, `host_id`, `host_thumbnail_url`, `host_picture_url` Estas variables son identificadores únicos y URLs que no aportan valor predictivo.
# MAGIC - **Información Redundante o de Identificación del Anfitrión:** `host_about`, `host_name`, `host_location`, `host_neighbourhood` Detalles personales del anfitrión que no son relevantes para la decisión de alquilar una propiedad.
# MAGIC - **Fechas de Scraping y Revisiones:** `last_scraped`, `host_since`, `calendar_last_scraped`, `first_review`, `last_review` Las fechas relacionadas con el scraping y las revisiones que no impactan las características de la propiedad o las decisiones del usuario.
# MAGIC - **Variables con Información Muy Específica:** `host_total_listings_count`, `minimum_minimum_nights`, `maximum_minimum_nights`, `minimum_maximum_nights`, `maximum_maximum_nights`, `minimum_nights_avg_ntm`, `maximum_nights_avg_ntm`, `bathrooms_text`, `host_response_time` Estas métricas detalladas sobre las políticas de reserva son redundantes o demasiado específicas para influir en el modelo general.

# COMMAND ----------

# Definición de columnas a eliminar
columns_to_drop = [
    'id', 'source', 'listing_url', 'scrape_id', 'host_id', 'host_url', 'picture_url', 'host_thumbnail_url', 
    'host_picture_url', 'name', 'host_about', 'host_name', 'host_location', 'host_neighbourhood', 
    'has_availability', 'availability_30', 'availability_60', 'availability_90', 'availability_365',
    'last_scraped', 'host_since', 'calendar_last_scraped', 'first_review', 'last_review', 'host_response_time',
    'host_total_listings_count', 'calculated_host_listings_count', 'calculated_host_listings_count_entire_homes', 'calculated_host_listings_count_private_rooms', 'calculated_host_listings_count_shared_rooms', 'minimum_minimum_nights', 'maximum_minimum_nights', 'bathrooms_text', 'minimum_maximum_nights', 'maximum_maximum_nights', 'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm'
]

# Eliminación de columnas
df = df.drop(*columns_to_drop)

# COMMAND ----------

# MAGIC %md
# MAGIC La variable `amenities` contiene la información de los servicios disponibles en las propiedades. Esta variables es una lista de elementos, pero resulta no estar normalizada por lo que hay muchos servicios únicos. Es por eso que se decide tomar los 30 amenities más comunes para una representación más eficiente y relevante en análisis posteriores.
# MAGIC
# MAGIC Realizar **OneHotEncoding** con estas 30 categorias aumentara demasiado la dimensionalidad de la base, por lo que se opta por reducir la dimensionalidad utilizando **PCA**, para identificar las principales componentes que capturan la mayor parte de la variabilidad en los datos.

# COMMAND ----------

# Explode la columna 'amenities' para trabajar con cada amenidad individualmente
df_exploded = df.withColumn("amenity", explode("amenities"))

# Contar ocurrencias y seleccionar las 30 amenities más comunes
top_amenities = df_exploded.groupBy("amenity").count().orderBy("count", ascending=False).limit(30)
top_amenities_list = [row['amenity'] for row in top_amenities.collect()]

# COMMAND ----------

# Visualización de las 30 amenities más comunes
pandas_df = top_amenities.toPandas()

plt.figure(figsize=(16, 10))
plt.barh(pandas_df['amenity'], pandas_df['count'], color='skyblue')
plt.xlabel('Cantidad')
plt.title('Top 30 Amenities')
plt.gca().invert_yaxis()  
plt.show()

# COMMAND ----------

# Crear una columna con un array de booleanos donde cada elemento indica la presencia de una amenity
@udf(ArrayType(IntegerType()))
def amenities_encoding(amenities):
    '''
    Función para encodear las amenities.
    - Input: lista de amenities a encodear
    - Output: Array de booleanos donde cada elemento indica la presencia de una amenity
    '''
    return [1 if amenity in amenities else 0 for amenity in top_amenities_list]

df = df.withColumn("amenities", amenities_encoding(col("amenities")))

# Convertir la lista de enteros en un vector denso
to_vector_udf = udf(lambda x: Vectors.dense(x), VectorUDT())
df = df.withColumn("amenities", to_vector_udf(col("amenities")))

# 1. Aplicar PCA para reducir la dimensionalidad
stages.append(PCA(k=10, inputCol="amenities", outputCol="pcaAmenities"))

# COMMAND ----------

# MAGIC %md
# MAGIC Para la variable `room_type`, que clasifica el tipo de habitación ofrecido, se utilizará la técnica de **One-Hot Encoding (OHE)**. Esta técnica es ideal para variables categóricas sin un orden inherente porque convierte cada categoría en una nueva columna binaria, permitiendo que los modelos traten cada tipo de habitación como una entidad separada sin asignar un orden arbitrario.
# MAGIC
# MAGIC En cuanto a la variable `property_type`, aunque proporciona información detallada sobre el tipo de propiedad, en muchos casos esta información es un refinamiento de `room_type` y agrega una complejidad innecesaria con sus 66 categorías únicas. Por lo tanto, se ha decidido eliminar property_type del análisis para simplificar el modelo y evitar la dilución de variables significativas.

# COMMAND ----------

# 2. One-Hot Encoding de 'room_type'
indexer = StringIndexer(inputCol="room_type", outputCol="room_type_Index")
encoder = OneHotEncoder(inputCols=["room_type_Index"], outputCols=["room_type_OHE"])
stages += [indexer, encoder]

# COMMAND ----------

# MAGIC %md
# MAGIC La columna `description` contiene texto extenso que describe la propiedad. Este texto puede incluir información valiosa sobre las características y ventajas de la propiedad que no están explícitamente listadas en otras columnas.
# MAGIC
# MAGIC Para la variable `description`, podemos utilizar técnicas de procesamiento de lenguaje natural (NLP) para extraer features útiles que puedan ser utilizadas por el modelo. Una técnica común es la vectorización TF-IDF (Term Frequency-Inverse Document Frequency), que nos permite convertir texto no estructurado en un formato estructurado, manteniendo la información sobre la importancia de las palabras en el texto.

# COMMAND ----------

# 3. Tokenización: Dividir el texto en palabras
tokenizer = Tokenizer(inputCol="description", outputCol="words")

# 4. Hashing TF: Convertir array de palabras a vectores de frecuencia de términos
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20)

# 5. IDF: Computar el Inverse Document Frequency para evaluar la importancia de una palabra en los documentos
idf = IDF(inputCol="rawFeatures", outputCol="idfFeatures")

# Agregar stages al pipeline
stages += [tokenizer, hashingTF, idf]

# COMMAND ----------

# MAGIC %md
# MAGIC En el análisis de propiedades geográficas para nuestro estudio multiciudad, utilizamos las variables de `latitude` y `longitude` para estimar las distancias de cada propiedad a un punto de interés central en cada ciudad. Este enfoque permite una evaluación directa y comparable de la ubicación sin depender de descripciones subjetivas como `neighborhood_overview` y `neighbourhood_cleansed`.
# MAGIC
# MAGIC Hemos seleccionado un punto de interés central en cada una de las ciudades analizadas debido a su importancia cultural, histórica, o turística, que se espera tenga un impacto significativo en la demanda de alojamiento:
# MAGIC
# MAGIC - **Buenos Aires:** El Obelisco, símbolo icónico y central.
# MAGIC - **Londres:** Trafalgar Square, punto clave y turístico.
# MAGIC - **París:** Notre Dame, ubicación céntrica y de gran atractivo turístico.
# MAGIC - **Roma:** Coliseo, emblemático y frecuentemente visitado.
# MAGIC - **Nueva York:** Times Square, corazón cultural y turístico.
# MAGIC
# MAGIC Para cada propiedad, calcularemos la **distancia Manhattan** desde el punto central seleccionado. Esta métrica es particularmente útil para ciudades donde las rutas rectas son menos comunes debido a la estructura urbana. La distancia Manhattan ofrece una medida de la accesibilidad a estos puntos centrales, y se espera que proporcione una correlación con la demanda de las propiedades en Airbnb.
# MAGIC
# MAGIC Este análimen proporciona una base cuantificable para comparar propiedades a través de diversas ubicaciones urbanas, facilitando un entendimiento más claro de cómo la proximidad a áreas de alto interés puede influenciar la valoración y la ocupación de un alojamiento en Airbnb.

# COMMAND ----------

# Definir los centros de cada ciudad en términos de latitud y longitud
centros = {
    "Buenos Aires": (-34.6037, -58.3816), # El Obelisco
    "London": (51.5079, -0.1281), # Trafalgar Square
    "Paris": (48.8527, 2.3508), # Notre Dame
    "Rome": (41.8902, 12.4922), # Coliseo
    "New York": (40.7580, -73.9855) # Times Square
}

# Definir UDF para calcular la distancia Manhattan
def manhattan_distance(lat, lon, city):
    '''
    Función para calcular la distancia Manhattan.
    - Input: latitud, longitud y ciudad
    - Output: distancia manhattan
    '''
    centro_lat, centro_lon = centros[city]
    delta_lat = lat - centro_lat
    delta_lon = lon - centro_lon
    return (delta_lat if delta_lat > 0 else -delta_lat) + (delta_lon if delta_lon > 0 else -delta_lon)

# Registrar UDF
distance_udf = udf(manhattan_distance, DoubleType())

# Aplicar la UDF al DataFrame
df = df.withColumn("distancia_centro", distance_udf(col("latitude"), col("longitude"), col("city")))

# COMMAND ----------

# MAGIC %md
# MAGIC Para mejorar la contribución de la variable `price`  y asegurar que no domine debido a su rango de valores, se aplicará una normalización con **MinMaxScaler**. Esta normalización transformará todos los valores de `price` a una escala entre 0 y 1, donde 0 representa el precio mínimo y 1 el precio máximo observado en los datos de entrenamiento. 

# COMMAND ----------

# 6. Crear un VectorAssembler para convertir 'price' en un vector
assembler = VectorAssembler(inputCols=["price"], outputCol="price_vec")

# 7. Configurar el MinMaxScaler para normalizar 'price'
scaler = MinMaxScaler(inputCol="price_vec", outputCol="price_scaled")

# Añadir ambos, el assembler y el scaler, a las etapas del pipeline
stages += [assembler, scaler]

# COMMAND ----------

# Se identifican las variables que seran utilizadas. 
feature_corpus = ['host_response_rate', 'host_acceptance_rate', 'host_listings_count', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'minimum_nights', 'maximum_nights', 'number_of_reviews', 'number_of_reviews_ltm', 'number_of_reviews_l30d', 'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value', 'reviews_per_month', 'verif_phone', 'verif_photographer', 'verif_email', 'verif_work_email', 'distancia_centro', 'pcaAmenities', 'host_is_superhost', 'host_has_profile_pic', 'host_identity_verified', 'instant_bookable', 'room_type_OHE', 'idfFeatures', 'price_scaled']

# 8. Crear un VectorAssembler para crear el corpus de Features
assemblerFeatures = VectorAssembler(inputCols=feature_corpus, outputCol="features")
stages.append(assemblerFeatures)

# COMMAND ----------

# Verificacion del Pipeline
partialPipeline = Pipeline().setStages(stages)
partialPipeline.getStages()

# COMMAND ----------

# Prueba Pipeline
pipelineModel = partialPipeline.fit(df)
preppedDataDF = pipelineModel.transform(df)

display(preppedDataDF)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Split de Datos
# MAGIC Para entrenar el modelo de forma efectiva y poder evaluarlos correctamente, se dividen los datos en conjuntos de entrenamiento y prueba. Esta división ayuda a evitar el sobreajuste y garantiza que el modelo pueda generalizar bien a nuevos datos. Se utiliza una proporción 80/20.
# MAGIC
# MAGIC El conjunto de entrenamiento se utiliza para ajustar los parámetros del modelo, mientras que el conjunto de prueba se utiliza para evaluar la performance del modelo en datos que no ha visto durante el entrenamiento.
# MAGIC

# COMMAND ----------

# Dividir los datos en conjuntos de entrenamiento (70%) y prueba (30%)
train_df, test_df = df.randomSplit([0.7, 0.3], seed=42)

# Mostrar los tamaños de cada DataFrame para verificar la división
print(f"Número de filas en el conjunto de entrenamiento: {train_df.count()}")
print(f"Número de filas en el conjunto de prueba: {test_df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Creación de Base con Feature Engineering
# MAGIC Se crea la base final de los features que se utilizaran en el modelo. 

# COMMAND ----------

# Se hace el fit en la base de train
Pipeline = partialPipeline.fit(train_df)

# Se transforman las bases de train y test
train_df = Pipeline.transform(train_df)
test_df = Pipeline.transform(test_df)

# COMMAND ----------

# Escritura de las bases de train y test
train_df.select('features', 'city','occupation_rate').write.mode("overwrite").parquet("/dbfs/FileStore/FeaturesTrain.parquet")
test_df.select('features', 'city','occupation_rate').write.mode("overwrite").parquet("/dbfs/FileStore/FeaturesTest.parquet")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Modelo de Machine Learning
# MAGIC ### Objetivo
# MAGIC Desarrollar un modelo predictivo utilizando Spark que permita a los anfitriones de Airbnb en varias ciudades predecir la ocupación de sus propiedades basándose en datos históricos y tendencias de mercado. El objetivo es utilizar los insights generados para recomendar estrategias de precios que maximicen la ocupación y, por ende, los ingresos.
# MAGIC
# MAGIC ### Configuración de MLflow
# MAGIC Se utiliza MLflow para realizar el seguimiento de los experimentos y comparar los resultados de diferentes configuraciones del modelo. Esto permite una evaluación eficiente y sistemática de las mejoras en el modelo.
# MAGIC
# MAGIC ### Definición del Modelo y Pipeline
# MAGIC Se definen modelos de XGBoost y RandomForest y se construye un pipeline en Spark que incluye todas las etapas de preprocesamiento y el propio modelo.

# COMMAND ----------

# Lectura de las bases de train y test
train_df = spark.read.parquet("/dbfs/FileStore/FeaturesTrain.parquet")
test_df = spark.read.parquet("/dbfs/FileStore/FeaturesTest.parquet")

# COMMAND ----------

# Evaluador común para ambos modelos
rmse_evaluator = RegressionEvaluator(labelCol="occupation_rate", predictionCol="prediction", metricName="rmse")
mae_evaluator = RegressionEvaluator(labelCol="occupation_rate", predictionCol="prediction", metricName="mae")
r2_evaluator = RegressionEvaluator(labelCol="occupation_rate", predictionCol="prediction", metricName="r2")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Configuración de MLFlow

# COMMAND ----------

# Función para ejecutar y registrar un modelo
def run_mlflow_experiment(pipeline, model_name):
    '''
    Función para ejecutar y registrar un modelo
    - Input: pipeline a ejecutar y nombre del modelo
    '''
    mlflow.set_experiment(f"/Users/anoguera@itba.edu.ar/Airbnb_Occupation_{model_name}")
    with mlflow.start_run(run_name=f"{model_name} Model"):
        # Ajustar el modelo
        model = pipeline.fit(train_df)

        # Predicciones
        predictions = model.transform(test_df)

        # Evaluar
        rmse = rmse_evaluator.evaluate(predictions)
        mae = mae_evaluator.evaluate(predictions)
        r2 = r2_evaluator.evaluate(predictions)
        
        y_true = [row['occupation_rate'] for row in predictions.select("occupation_rate").collect()]
        y_pred = [row['prediction'] for row in predictions.select("prediction").collect()]
        
        # Calcular métricas adicionales
        mse = mean_squared_error(y_true, y_pred)

        # Log metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)

        # Log Model
        mlflow.log_param("model_name", model_name)
        mlflow.spark.log_model(model, "model")
        
        # Residual Plot
        residuals = [y_t - y_p for y_t, y_p in zip(y_true, y_pred)]
        plt.figure()
        sns.histplot(residuals, kde=True)
        plt.title("Residuals")
        plt.savefig("residuals.png")
        mlflow.log_artifact("residuals.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Experimento 1: XGBoost

# COMMAND ----------

# Configurar Modelo
xgboost = GBTRegressor(featuresCol='features', labelCol='occupation_rate')

# Configurar grid de hiperpárametros
paramGrid_xgb = ParamGridBuilder() \
    .addGrid(xgboost.maxDepth, [5, 10, 15]) \
    .addGrid(xgboost.stepSize, [0.05, 0.01, 0.1, 0.2]) \
    .addGrid(xgboost.maxIter, [50, 100, 150]) \
    .addGrid(xgboost.subsamplingRate, [0.8, 0.9, 1.0]) \
    .build()

# Evaluator
evaluator = RegressionEvaluator(labelCol='occupation_rate', predictionCol='prediction', metricName='rmse')

# Crear y configurar CrossValidator
crossval_xgb = CrossValidator(estimator=xgboost,
                              estimatorParamMaps=paramGrid_xgb,
                              evaluator=evaluator,
                              numFolds=5)

# Definir el Pipeline completo con CrossValidator
pipeline_xgb = Pipeline(stages=[crossval_xgb])

# COMMAND ----------

# Ejecutar y registrar para XGBoost
run_mlflow_experiment(pipeline_xgb, "XGBoost")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Experimento 2: RandomForest

# COMMAND ----------

# Configurar Modelo
randomForest = RandomForestRegressor(featuresCol='features', labelCol='occupation_rate')

# Configurar grid de hiperparámetros
paramGrid_rf = ParamGridBuilder() \
    .addGrid(randomForest.numTrees, [20, 50]) \
    .addGrid(randomForest.maxDepth, [5, 10]) \
    .build()

# Crear y configurar CrossValidator
crossval_rf = CrossValidator(estimator=randomForest,
                             estimatorParamMaps=paramGrid_rf,
                             evaluator=rmse_evaluator,
                             numFolds=5)

# Definir el Pipeline completo
pipeline_rf = Pipeline(stages=[crossval_rf])

# COMMAND ----------

# Ejecutar y registrar para RandomForest
run_mlflow_experiment(pipeline_rf, "RandomForest")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluación de los Modelos de Regresión
# MAGIC En este análisis, se evaluó el rendimiento de varios modelos de regresión para predecir la tasa de ocupación. Utilizamos validación cruzada para optimizar los hiperparámetros y registramos las métricas de rendimiento en mlflow. A continuación, se presentan los resultados de las métricas clave, como RMSE, MAE y R², junto con gráficos comparativos, incluyendo el Gráfico de Residuos, Gráfico de Predicciones vs Valores Reales y Gráfico de Predicciones vs Residuos, para identificar el modelo más eficiente y robusto.

# COMMAND ----------

def get_model_metrics(run_id):
    '''
    Función para recuperar métricas de mlflow
    - Input: identificación de corrida. 
    - Output: métricas de la corrida.
    '''
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    metrics = run.data.metrics
    params = run.data.params
    return {
        "model_name": params.get("model_name"),
        "rmse": metrics.get("rmse"),
        "mae": metrics.get("mae"),
        "mse": metrics.get("mse"),
        "r2": metrics.get("r2")
    }

def display_mlflow_artifacts_grid(run_ids):
    '''
    Función para mostrar gráficos de mlflow en una cuadrícula
    - Input: lista de run_ids.
    '''
    num_runs = len(run_ids)
    fig = plt.figure(figsize=(19, 3 * num_runs))
    gs = GridSpec(1, num_runs, figure=fig)

    for i, run_id in enumerate(run_ids):
        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(run_id)
        artifact_paths = {artifact.path: artifact for artifact in artifacts}
        run = client.get_run(run_id)
        params = run.data.params
        model_name = params.get("model_name")
        
        if "residuals.png" in artifact_paths:
            local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="residuals.png")
            img = plt.imread(local_path)
            ax = fig.add_subplot(gs[0, i])
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f"Residuals\nModel Name: {model_name}")

    plt.tight_layout()
    plt.show()


# COMMAND ----------

# Definir los ids de corridas
run_ids = ["e2e80600ee6d45dcb07ae0e383778ba2", "f3c2b6a1affd4db98abb44703664620f"] 

# COMMAND ----------

# Recuperar métricas de todos los modelos
model_metrics = [get_model_metrics(run_id) for run_id in run_ids]

# Convertir a DataFrame para visualización
metrics_df = pd.DataFrame(model_metrics)

# Mostrar métricas comparativas
metrics_df.display()

# COMMAND ----------

# Mostrar gráficos en una cuadrícula para cada run
display_mlflow_artifacts_grid(run_ids)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Interpretación de Resultados
# MAGIC
# MAGIC En la evaluación de los modelos para predecir la tasa de ocupación, se probaron dos algoritmos: XGBoost y Random Forest. A continuación, se presenta una interpretación resumida de los resultados obtenidos:
# MAGIC
# MAGIC - **XGBoost:** El modelo de XGBoost muestra una capacidad limitada para predecir la tasa de ocupación. El RMSE y MAE indican un error moderado en las predicciones. El valor de R² de 0.211280 sugiere que solo una pequeña proporción de la varianza en la tasa de ocupación está siendo explicada por este modelo. El gráfico de residuos muestra una distribución dispersa, lo que indica variabilidad significativa en los errores del modelo.
# MAGIC
# MAGIC - **Random Forest:** El modelo de Random Forest presenta un rendimiento ligeramente mejor que el XGBoost. Con un RMSE y MAE ligeramente menores, este modelo muestra una mejor capacidad de predicción. El R² de 0.240937 indica una mayor proporción de varianza explicada en comparación con el XGBoost, aunque sigue siendo limitada. El gráfico de residuos de Random Forest también muestra variabilidad, pero ligeramente mejor distribuida que en el modelo de XGBoost.
# MAGIC
# MAGIC ### Conclusión
# MAGIC
# MAGIC A partir de los resultados obtenidos, se puede observar que los valores de RMSE, MAE y R² de ambos modelos (XGBoost y Random Forest) no son óptimos, sugiriendo una baja capacidad predictiva de los modelos para la tasa de ocupación. Los valores de R² cercanos a cero indican que una pequeña proporción de la varianza en la tasa de ocupación está siendo explicada por los modelos.
# MAGIC
# MAGIC Dado esto, se propone cambiar la modalidad del análisis de predicción a clasificación. Específicamente, se buscará predecir si la tasa de ocupación de un establecimiento será mayor o menor que la media de su ciudad.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Modelo de Machine Learning
# MAGIC
# MAGIC ### Objetivo
# MAGIC Desarrollar un modelo predictivo utilizando Spark que permita a los anfitriones de Airbnb en varias ciudades predecir si la ocupación de sus propiedades será mayor o menor que la media de su ciudad, basándose en datos históricos y tendencias de mercado. El objetivo es utilizar los insights generados para recomendar estrategias de precios que maximicen la ocupación y, por ende, los ingresos.
# MAGIC
# MAGIC ### Configuración de MLflow
# MAGIC Se utiliza MLflow para realizar el seguimiento de los experimentos y comparar los resultados de diferentes configuraciones del modelo. Esto permite una evaluación eficiente y sistemática de las mejoras en el modelo.
# MAGIC
# MAGIC ### Definición del Modelo y Pipeline
# MAGIC Se definen modelos de XGBoost y RandomForest y se construye un pipeline en Spark que incluye todas las etapas de preprocesamiento y el propio modelo. La nueva modalidad se enfoca en la clasificación, prediciendo si la occupation_rate de un establecimiento será mayor o menor que la media de su ciudad.

# COMMAND ----------

# Calcular el promedio de occupation_rate para cada city en el conjunto de entrenamiento
avg_occupation_rate_df = train_df.groupBy("city").agg(avg("occupation_rate").alias("avg_occupation_rate"))

# Unir el promedio al conjunto de datos de entrenamiento y prueba
train_df = train_df.join(avg_occupation_rate_df, on="city", how="left")
test_df = test_df.join(avg_occupation_rate_df, on="city", how="left")

# Definir la variable occupation_success
train_df = train_df.withColumn("occupation_success", when(col("occupation_rate") > col("avg_occupation_rate"), 1).otherwise(0))
test_df = test_df.withColumn("occupation_success", when(col("occupation_rate") > col("avg_occupation_rate"), 1).otherwise(0))

# COMMAND ----------

# Definir modelos y grid search
models = {
    "RandomForest": RandomForestClassifier(labelCol="occupation_success", featuresCol="features"),
    "LogisticRegression": LogisticRegression(labelCol="occupation_success", featuresCol="features"),
    "DecisionTree": DecisionTreeClassifier(labelCol="occupation_success", featuresCol="features")
}

param_grids = {
    "RandomForest": ParamGridBuilder().addGrid(models["RandomForest"].numTrees, [10, 20]).build(),
    "LogisticRegression": ParamGridBuilder().addGrid(models["LogisticRegression"].regParam, [0.01, 0.1]).build(),
    "DecisionTree": ParamGridBuilder().addGrid(models["DecisionTree"].maxDepth, [5, 10]).build()
}


evaluator = BinaryClassificationEvaluator(labelCol="occupation_success")

# COMMAND ----------

def train_and_log_model(model_name, model, param_grid):
    '''
    Función para entrenar y registrar modelos en MLflow
    - Input: nombre del modelo, modelo y grilla de parametros. 
    '''
    with mlflow.start_run(run_name=model_name):
        crossval = CrossValidator(estimator=model,
                                  estimatorParamMaps=param_grid,
                                  evaluator=evaluator,
                                  numFolds=3)
        
        cv_model = crossval.fit(train_df)
        best_model = cv_model.bestModel

        # Evaluar el modelo en el conjunto de test
        predictions = best_model.transform(test_df)
        
        # Obtener métricas de evaluación
        roc_auc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})
        pr_auc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderPR"})
        
        # Obtener las predicciones y etiquetas verdaderas
        y_true = [row['occupation_success'] for row in predictions.select("occupation_success").collect()]
        y_pred = [row['prediction'] for row in predictions.select("prediction").collect()]
        
        # Calcular métricas adicionales
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        
        # Registrar métricas en mlflow
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("best_params", best_model.extractParamMap())
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("pr_auc", pr_auc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("accuracy", accuracy)
        
        # ROC Curve
        y_scores = [row.probability[1] for row in predictions.select("probability").collect()]
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        plt.figure()
        plt.plot(fpr, tpr)
        plt.title("ROC Curve")
        plt.savefig("roc_curve.png")
        mlflow.log_artifact("roc_curve.png")

        # Precision-Recall Curve
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_scores)
        plt.figure()
        plt.plot(recall_curve, precision_curve)
        plt.title("Precision-Recall Curve")
        plt.savefig("pr_curve.png")
        mlflow.log_artifact("pr_curve.png")

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure()
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title("Confusion Matrix")
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")

        # Residual Plot
        residuals = [y_t - y_p for y_t, y_p in zip(y_true, y_pred)]
        plt.figure()
        sns.histplot(residuals, kde=True)
        plt.title("Residuals")
        plt.savefig("residuals.png")
        mlflow.log_artifact("residuals.png")


# COMMAND ----------

# Entrenar y registrar modelos
for model_name, model in models.items():
    train_and_log_model(model_name, model, param_grids[model_name])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluación de los Modelos de Clasificación
# MAGIC En este análisis, se evaluó el rendimiento de varios modelos de clasificación para predecir la ocupación exitosa. Utilizamos validación cruzada para optimizar los hiperparámetros y registramos las métricas de rendimiento en mlflow. A continuación, se presentan los resultados de las métricas clave, como ROC AUC, PR AUC, Accuracy, Precision y Recall, junto con gráficos comparativos, incluyendo la Curva ROC, Curva de Precisión-Recall, Matriz de Confusión y Gráfico de Residuales, para identificar el modelo más eficiente y robusto.

# COMMAND ----------

def get_model_metrics(run_id):
    '''
    Función para recuperar métricas de mlflow
    - Input: identificación de corrida. 
    - Output: métricas de la corrida.
    '''
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    metrics = run.data.metrics
    params = run.data.params
    return {
        "model_name": params.get("model_name"),
        "roc_auc": metrics.get("roc_auc"),
        "pr_auc": metrics.get("pr_auc"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "accuracy": metrics.get("accuracy"),
    }

def display_mlflow_artifacts_grid(run_ids):
    '''
    Función para mostrar gráficos de mlflow en una cuadrícula
    - Input: identificación de corrida. 
    '''
    num_runs = len(run_ids)
    fig = plt.figure(figsize=(19, 3 * num_runs))
    gs = GridSpec(num_runs, 4, figure=fig)

    for i, run_id in enumerate(run_ids):
        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(run_id)
        artifact_paths = {artifact.path: artifact for artifact in artifacts}
        run = client.get_run(run_id)
        params = run.data.params
        model_name = params.get("model_name")
        
        if "roc_curve.png" in artifact_paths:
            local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="roc_curve.png")
            img = plt.imread(local_path)
            ax = fig.add_subplot(gs[i, 0])
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f"ROC Curve\nModel Name: {model_name}")
        
        if "pr_curve.png" in artifact_paths:
            local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="pr_curve.png")
            img = plt.imread(local_path)
            ax = fig.add_subplot(gs[i, 1])
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f"Precision-Recall Curve\nModel Name: {model_name}")
        
        if "confusion_matrix.png" in artifact_paths:
            local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="confusion_matrix.png")
            img = plt.imread(local_path)
            ax = fig.add_subplot(gs[i, 2])
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f"Confusion Matrix\nModel Name: {model_name}")
        
        if "residuals.png" in artifact_paths:
            local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="residuals.png")
            img = plt.imread(local_path)
            ax = fig.add_subplot(gs[i, 3])
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f"Residuals\nModel Name: {model_name}")

    plt.tight_layout()
    plt.show()

# COMMAND ----------

# Definir los ids de corridas
run_ids = ["93cb2f81d9274ab8b3ca419e4926dfd7", "8e36fc81912e4c4ca9bd101b7903341a", "7f6ddcb2a6df4eea82229dd2c5dace9f"] 

# COMMAND ----------

# Recuperar métricas de todos los modelos
model_metrics = [get_model_metrics(run_id) for run_id in run_ids]

# Convertir a DataFrame para visualización
metrics_df = pd.DataFrame(model_metrics)

# Mostrar métricas comparativas
metrics_df.display()

# COMMAND ----------

# Mostrar gráficos en una cuadrícula para cada run
display_mlflow_artifacts_grid(run_ids)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Interpretación de Resultados
# MAGIC
# MAGIC En la evaluación de los modelos para predecir la ocupación exitosa, se probaron tres algoritmos: Decision Tree, Logistic Regression y Random Forest. A continuación, se presenta una interpretación resumida de los resultados obtenidos:
# MAGIC
# MAGIC - **Decision Tree:** Este modelo obtuvo un ROC AUC de 0.613 y un PR AUC de 0.606. Estos valores indican una capacidad moderada para distinguir entre clases. Los gráficos muestran una curva ROC que apenas se despega de la línea de azar, y una curva de Precisión-Recall que refleja un rendimiento limitado. La matriz de confusión revela una cantidad considerable de falsos positivos y falsos negativos, mientras que el gráfico de residuos indica una variabilidad significativa en los errores del modelo. Con una precisión de 0.619, un recall de 0.611 y una exactitud de 0.607, este modelo presenta un rendimiento limitado en términos de discriminación y balance entre precisión y recall.
# MAGIC
# MAGIC - **Logistic Regression:** Este modelo presentó un mejor rendimiento comparado con el Decision Tree, obteniendo un ROC AUC de 0.665 y un PR AUC de 0.665. La curva ROC es más pronunciada, indicando una mejor capacidad de discriminación. La curva de Precisión-Recall sugiere un buen balance entre precisión y recall. La matriz de confusión muestra una reducción en los errores de clasificación comparado con el Decision Tree, y el gráfico de residuos indica una distribución más centrada alrededor de cero, sugiriendo menos sesgo. Con una precisión de 0.619, un recall de 0.673 y una exactitud de 0.619, este modelo mejora en todos los aspectos clave sobre el Decision Tree.
# MAGIC
# MAGIC - **Random Forest:** Este modelo fue el mejor de los tres, con un ROC AUC de 0.668 y un PR AUC de 0.673. La curva ROC es la más alta entre los modelos evaluados, demostrando la mejor capacidad de discriminación. La curva de Precisión-Recall también es superior, indicando que el modelo maneja eficazmente el balance entre precisión y recall. La matriz de confusión revela una disminución en los falsos positivos y negativos, y el gráfico de residuos muestra una distribución adecuada alrededor de cero, sugiriendo un buen ajuste del modelo sin sesgo significativo. Con una precisión de 0.610, un recall de 0.704 y una exactitud de 0.617, este modelo demuestra ser el más eficiente y robusto para la tarea de clasificación de ocupación exitosa.
# MAGIC
# MAGIC ### Conclusión
# MAGIC
# MAGIC De los tres modelos evaluados, el **Random Forest** se destaca como el más eficiente y robusto para la tarea de clasificación de ocupación exitosa. Presenta las mejores métricas de rendimiento (ROC AUC y PR AUC) y un rendimiento superior en los gráficos comparativos, superando claramente a los modelos de Decision Tree y Logistic Regression.
# MAGIC

# COMMAND ----------

