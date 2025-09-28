# news_streaming.py
import os
import pandas as pd
import joblib
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import StructType, StructField, StringType

# ------------------------------
# Configuration
# ------------------------------
RAW_FOLDER = "news_stream"
PROCESSED_FOLDER = "processed_news"
CHECKPOINT_FOLDER = "checkpoints"
MODEL_PATH = "model.pkl"

os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(CHECKPOINT_FOLDER, exist_ok=True)

# ------------------------------
# Load scikit-learn model
# ------------------------------
model = joblib.load(MODEL_PATH)

# ------------------------------
# Start Spark session
# ------------------------------
spark = SparkSession.builder.appName("NewsSentimentStreaming").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# ------------------------------
# Define schema
# ------------------------------
schema = StructType([
    StructField("text", StringType(), True),
    StructField("publishedAt", StringType(), True),
    StructField("source", StringType(), True),
    StructField("category", StringType(), True)
])

# ------------------------------
# Define Pandas UDF for sentiment
# ------------------------------
@pandas_udf("string", PandasUDFType.SCALAR)
def predict_sentiment(text_series: pd.Series) -> pd.Series:
    preds = model.predict(text_series.fillna("").astype(str))
    return pd.Series(["✅ Positive" if p == 1 else "❌ Negative" for p in preds])

# ------------------------------
# Read streaming data from CSV folder
# ------------------------------
df_stream = spark.readStream \
    .option("header", "true") \
    .schema(schema) \
    .csv(RAW_FOLDER)

# Apply sentiment prediction
df_with_sentiment = df_stream.withColumn("sent_label", predict_sentiment(df_stream["text"]))

# ------------------------------
# Write stream to processed_news folder
# ------------------------------
query = df_with_sentiment.writeStream \
    .outputMode("append") \
    .format("csv") \
    .option("checkpointLocation", CHECKPOINT_FOLDER) \
    .option("path", PROCESSED_FOLDER) \
    .trigger(processingTime="10 seconds") \
    .start()

query.awaitTermination()
