from pyspark.sql import SparkSession
from pyspark.sql.functions import when,col
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import StandardScaler,VectorAssembler,PCA
import matplotlib.pyplot as plt

spark = SparkSession.builder.appName('Clustering').getOrCreate()
df_spark = spark.read.csv("/home/soham/BigData/datasets/segmentation_data.csv",header=True,inferSchema=True)
df_spark.columns
features =[
 'Sex',
 'Marital status',
 'Age',
 'Education',
 'Income',
 'Occupation',
 'Settlement size']
assembler = VectorAssembler(inputCols=features,outputCol="vectorized_features")
vectorized_df = assembler.transform(df_spark)
scale = StandardScaler(inputCol="vectorized_features",outputCol="scaled_features")
scaler = scale.fit(vectorized_df)
scaled_df = scaler.transform(vectorized_df)
pca = PCA(k=3, inputCol="scaled_features", outputCol="pca_features")
pca_model = pca.fit(scaled_df)
pca_result = pca_model.transform(scaled_df)

# Create Clusters

cluster = KMeans(featuresCol="pca_features", k=6)
model = cluster.fit(pca_result)
predictions = model.transform(pca_result)
predictions.show(10)

# Plot Clusters

pca_plot = predictions.select(["pca_features","prediction"]).toPandas()
pca_plot['PC1']= pca_plot["pca_features"].apply(lambda x:x[0])
pca_plot["PC2"] = pca_plot["pca_features"].apply(lambda x:x[1])
plt.scatter(pca_plot['PC1'], pca_plot['PC2'], c=pca_plot['prediction'], cmap='viridis', s=50, alpha=0.6)