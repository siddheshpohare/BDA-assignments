from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("TS").getOrCreate()
df = spark.read.csv("/home/soham/BigData/TrafficPrediction.csv",header=True,inferSchema=True)
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import to_date,day,month,year,to_timestamp,sum as spark_sum

df=df.withColumn("Datetime",to_timestamp(df["Datetime"],'dd-MM-yyyy HH:mm'))
df = df.withColumn("Day",day(df["Datetime"]))
df = df.withColumn("Month",month(df["Datetime"]))
df = df.withColumn("Year",year(df["Datetime"]))
df = df.groupBy("Month","Year").agg(spark_sum("Count").alias("Monthly_Count"))
df.show()
columns = df.columns
assembler = VectorAssembler(inputCols=columns, outputCol="features")
df_vector = assembler.transform(df)

LR = LinearRegression(featuresCol="features", labelCol="Monthly_Count",regParam=0.1)

train, test = df_vector.randomSplit([0.8, 0.2])

# Fit the model to the training data
lr_model = LR.fit(train)

# Optional: Evaluate the model on test data
predictions = lr_model.transform(test)
predictions.show()