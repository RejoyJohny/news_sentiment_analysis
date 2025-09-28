from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF
from pyspark.sql.functions import pandas_udf, PandasUDFType
import pandas as pd
import joblib

# Start Spark session
spark = SparkSession.builder.appName("NewsSentimentStreaming").getOrCreate()

# Schema: adjust based on your CSV structure
schema = "title STRING, description STRING, source STRING"

# Read stream from folder
news_stream = spark.readStream.format("csv") \
    .option("header", "true") \
    .schema(schema) \
    .load("news_stream/")

# Text preprocessing
tokenizer = Tokenizer(inputCol="title", outputCol="words")
words_data = tokenizer.transform(news_stream)

remover = StopWordsRemover(inputCol="words", outputCol="filtered")
filtered_data = remover.transform(words_data)

hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=1000)
featurized_data = hashingTF.transform(filtered_data)

# Load pre-trained ML model (scikit-learn)
model = joblib.load("model.pkl")

@pandas_udf("string", PandasUDFType.SCALAR)
def predict_sentiment(title_series: pd.Series) -> pd.Series:
    return pd.Series(model.predict(title_series))

# Apply model
news_with_sentiment = featurized_data.withColumn("sentiment", predict_sentiment("title"))

# Output to console (for debugging) or to CSV folder for dashboard
query = news_with_sentiment.writeStream \
    .outputMode("append") \
    .format("console") \
    .start()

query.awaitTermination()
