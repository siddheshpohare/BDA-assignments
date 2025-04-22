from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

# Initialize Spark session
spark = SparkSession.builder.appName("StockPricePredictionWithMA").getOrCreate()

# Load CSV file into a DataFrame
df = spark.read.csv("/home/soham/BigData/AAPL_data.csv", header=True, inferSchema=True)

# Create a window spec to calculate the moving average over a 5-day period (adjust the window size as needed)
windowSpec = Window.orderBy("Date").rowsBetween(-4, 0)

# Add a column for the 5-day moving average of the 'Price'
df = df.withColumn("MovingAvg", avg("Price").over(windowSpec))

# Fill any null values in the moving average column (may occur for the first few rows)
df = df.fillna({"MovingAvg": 0})

# Select relevant columns
df = df.select("Date", "Price", "Open", "High", "Low", "Volume", "MovingAvg")

# Feature engineering: Use Open, High, Low, Volume, and MovingAvg as features
assembler = VectorAssembler(inputCols=["Open", "High", "Low", "Volume", "MovingAvg"], outputCol="features")
df = assembler.transform(df)

# Split the data into training and test sets
train_df, test_df = df.randomSplit([0.8, 0.2], seed=1234)

# Initialize the Linear Regression model
lr = LinearRegression(featuresCol="features", labelCol="Price")

# Create the pipeline
pipeline = Pipeline(stages=[assembler, lr])

# Train the model
model = pipeline.fit(train_df)

# Make predictions
predictions = model.transform(test_df)

# Evaluate the model
evaluator = RegressionEvaluator(labelCol="Price", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)

print(f"Root Mean Squared Error (RMSE) on test data = {rmse}")
