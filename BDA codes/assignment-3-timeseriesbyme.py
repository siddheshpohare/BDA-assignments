from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Step 1: Create Spark session
spark = SparkSession.builder.appName("TrafficAnalysis").getOrCreate()

# Step 2: Set legacy time parser to fix datetime error
spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")

# Step 3: Load CSV
df = spark.read.csv("/content/Traffic.csv", header=True, inferSchema=True)

# Step 4: Create full datetime using fixed format
df = df.withColumn("Datetime", to_timestamp(
    concat_ws(" ", lit("10-04-2024"), col("Time")),
    "dd-MM-yyyy hh:mm:ss a"
))

# Step 5: Extract hour
df = df.withColumn("Hour", hour("Datetime"))

# Step 6: Categorize traffic level based on Vehicles (customize thresholds if needed)
df = df.withColumn("TrafficLevel", when(col("Total") < 30, "Low")
                                  .when((col("Total") >= 30) & (col("Total") < 70), "Normal")
                                  .otherwise("Heavy"))

# Step 7: Index the label column
indexer = StringIndexer(inputCol="TrafficLevel", outputCol="label")
df = indexer.fit(df).transform(df)

# Step 8: Assemble features
assembler = VectorAssembler(inputCols=["Hour"], outputCol="features")
df = assembler.transform(df)

# Step 9: Split data into train/test
train, test = df.randomSplit([0.8, 0.2], seed=42)

# Step 10: Train a classifier
dt = DecisionTreeClassifier(featuresCol="features", labelCol="label")
model = dt.fit(train)

# Step 11: Predict on test data
predictions = model.transform(test)

# Step 12: Evaluate accuracy
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy:", accuracy)

# Optional: Show some predictions
predictions.select("Hour", "Total", "TrafficLevel", "prediction").show()