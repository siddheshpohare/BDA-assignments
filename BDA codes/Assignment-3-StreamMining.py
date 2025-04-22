from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext

conf = SparkConf().setAppName("DataStreamMining")
sc = SparkContext(conf=conf)
ssc = StreamingContext(sc, 1)  

lines = ssc.socketTextStream("localhost", 9999)

words = lines.flatMap(lambda line: line.split(" "))  # Split each line into words
word_counts = words.map(lambda word: (word, 1)).reduceByKey(lambda a, b: a + b)  # Count words

word_counts.pprint()

ssc.start()
ssc.awaitTermination()
