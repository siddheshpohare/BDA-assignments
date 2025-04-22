from pyspark.sql import SparkSession
from pyspark.ml.feature import StandardScaler,VectorAssembler,StringIndexer
from pyspark.sql.functions import col,when
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder.appName("MV").getOrCreate()
df_spark = spark.read.csv("/home/soham/BigData/datasets/Iris.csv",header=True,inferSchema=True)
# encoding 
indexer = StringIndexer(inputCol="Species",outputCol="Enocded_Species")
index = indexer.fit(dataset=df_spark)
indexed_df=index.transform(df_spark)
df_spark = df_spark.dropna()
# vectorizing and scaling
features = [
 'SepalLengthCm',
 'SepalWidthCm',
 'PetalLengthCm',
 'PetalWidthCm']
assembler = VectorAssembler(inputCols=features,outputCol="vectorized_features")
scaler = StandardScaler(inputCol="vectorized_features",outputCol="scaled_features")
vectorized_df = assembler.transform(indexed_df)
scale = scaler.fit(vectorized_df)
scaled_df = scale.transform(vectorized_df)
LR = LogisticRegression(featuresCol="scaled_features",labelCol="Enocded_Species")
train,test = scaled_df.randomSplit([0.8,0.2])
model=LR.fit(train)
predictions = model.transform(test)
original_labels = index.labels


evaluator = MulticlassClassificationEvaluator(labelCol="Enocded_Species", predictionCol="prediction", metricName="accuracy")

# Evaluate model
accuracy = evaluator.evaluate(predictions)
print(f"Accuracy: {accuracy}")

for i,j in enumerate(original_labels):
    predictions=predictions.withColumn("prediction",when(col("prediction")==i,j).otherwise(col("prediction")))
predictions.show()