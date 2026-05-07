#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Task 6 — Hyperparameter Tuning

Notebook ini mencakup:
- Load model terbaik dari hasil evaluasi
- Tuning dengan CrossValidator + ParamGridBuilder
- Evaluasi model setelah tuning
- Perbandingan sebelum vs sesudah tuning
- Feature Importance
- Simpan model final

Prasyarat: Jalankan 04_modeling.ipynb dan 05_evaluation.ipynb terlebih dahulu.
"""

# ============================================================================
# 1. Setup
# ============================================================================

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.regression import RandomForestRegressor, RandomForestRegressionModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.feature import VectorAssembler
import os
import sys

# Fix untuk "Python worker failed to connect back" & "Python was not found (Microsoft Store)"
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

_java17_opts = (
    "-Djava.security.manager=allow "
    "--add-opens=java.base/java.lang=ALL-UNNAMED "
    "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED "
    "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED "
    "--add-opens=java.base/java.io=ALL-UNNAMED "
    "--add-opens=java.base/java.net=ALL-UNNAMED "
    "--add-opens=java.base/java.nio=ALL-UNNAMED "
    "--add-opens=java.base/java.util=ALL-UNNAMED "
    "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED "
    "--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED "
    "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED "
    "--add-opens=java.base/sun.nio.cs=ALL-UNNAMED "
    "--add-opens=java.base/sun.security.action=ALL-UNNAMED "
    "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED "
    "--add-opens=java.base/javax.security.auth=ALL-UNNAMED "
    "--add-opens=java.security.jgss/sun.security.krb5=ALL-UNNAMED"
)
os.environ["JAVA_TOOL_OPTIONS"] = _java17_opts

spark = SparkSession.builder \
    .appName("NYC_Taxi_Task6_Tuning") \
    .config("spark.driver.memory", "4g") \
    .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
print("SparkSession aktif:", spark.version)

# ============================================================================
# 2. Load Hasil Evaluasi Sebelum Tuning
# ============================================================================

# Load hasil evaluasi sebelumnya untuk perbandingan
try:
    eval_before_spark = spark.read.parquet("../outputs/evaluation_results.parquet")
    eval_before_spark.createOrReplaceTempView("eval_results")
    
    best_result = spark.sql("SELECT * FROM eval_results ORDER BY RMSE LIMIT 1").collect()[0]
    best_name = best_result["Model"]
    
    print(f"Model terbaik dari evaluasi sebelumnya: {best_name}")
    eval_before_spark.show()
    
    # Store results for later comparison
    all_results = {row["Model"]: (row["RMSE"], row["MAE"], row["R2"]) for row in eval_before_spark.collect()}
except Exception as e:
    print(f"⚠️  Error loading evaluation results: {e}")
    print("💡 Pastikan 05_evaluation.py telah dijalankan terlebih dahulu")
    spark.stop()
    exit(1)

# ============================================================================
# 3. Load Data untuk Training
# ============================================================================

print("\nMemuat data untuk tuning dan testing...")

EDA_DATA_PATH = "../data/raw/nyc_taxi_data.csv"
PROCESSED_DATA_PATH = "../data/processed/train.csv"

if os.path.exists(PROCESSED_DATA_PATH):
    df = spark.read.csv(PROCESSED_DATA_PATH, header=True, inferSchema=True)
else:
    df = spark.read.csv(EDA_DATA_PATH, header=True, inferSchema=True)
    df = df.withColumn("pickup_datetime", F.to_timestamp(F.col("pickup_datetime"), "yyyy-MM-dd HH:mm:ss z"))

df.createOrReplaceTempView("taxi_data_temp")
df = spark.sql("""
    SELECT 
        key, fare_amount, pickup_datetime,
        pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude,
        passenger_count,
        (2 * 6371.0 * asin(sqrt(
            sin(radians((dropoff_latitude - pickup_latitude) / 2)) * 
            sin(radians((dropoff_latitude - pickup_latitude) / 2)) +
            cos(radians(pickup_latitude)) * cos(radians(dropoff_latitude)) * 
            sin(radians((dropoff_longitude - pickup_longitude) / 2)) * 
            sin(radians((dropoff_longitude - pickup_longitude) / 2))
        ))) AS distance_km,
        hour(pickup_datetime) AS hour,
        dayofweek(pickup_datetime) AS day_of_week,
        month(pickup_datetime) AS month,
        year(pickup_datetime) AS year
    FROM taxi_data_temp
""")

FEATURE_COLS = [
    "distance_km", "hour", "day_of_week", "month", "year",
    "passenger_count", "pickup_longitude",  "pickup_latitude",
    "dropoff_longitude", "dropoff_latitude"
]

assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features")
df_model = assembler.transform(df).select("features", F.col("fare_amount").alias("label"))

# Gunakan seed dan SAMPLE_SIZE yang sama dengan 04_modeling.py
total_rows = df_model.count()
SAMPLE_SIZE = 10000
sample_fraction = min(SAMPLE_SIZE / total_rows, 1.0)
df_model = df_model.sample(fraction=sample_fraction, seed=42)

train_df, test_df = df_model.randomSplit([0.8, 0.2], seed=42)

print(f" Train set: {train_df.count():,} baris siap untuk tuning.")
print(f" Test set : {test_df.count():,} baris siap untuk evaluasi.")

# ============================================================================
# 4. Hyperparameter Tuning — Random Forest
# ============================================================================

print("\n" + "="*70)
print("Hyperparameter Tuning — Random Forest")
print("="*70)

# Definisi model & parameter grid
rf = RandomForestRegressor(featuresCol="features", labelCol="label", seed=42)

param_grid = ParamGridBuilder() \
    .addGrid(rf.numTrees,  [50, 100, 150]) \
    .addGrid(rf.maxDepth,  [5, 10, 15]) \
    .build()

evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")

cv = CrossValidator(
    estimator=rf,
    estimatorParamMaps=param_grid,
    evaluator=evaluator,
    numFolds=3,
    seed=42
)

print(f"Total kombinasi yang diuji: {len(param_grid)} kombinasi x 3 fold = {len(param_grid)*3} iterasi")
print("Memulai Cross Validation... (proses ini butuh beberapa menit)")

try:
    cv_model   = cv.fit(train_df)
    best_model = cv_model.bestModel

    print(f"\n Tuning selesai!")
    print(f"\nParameter terbaik:")
    print(f"  numTrees : {best_model.getNumTrees}")
    print(f"  maxDepth : {best_model.getOrDefault('maxDepth')}")
except Exception as e:
    print(f"  Tuning failed: {e}")
    print(" Menggunakan model default tanpa tuning")
    best_model = rf.fit(train_df)

# ============================================================================
# 5. Evaluasi Setelah Tuning
# ============================================================================

print("\n" + "="*70)
print("Evaluasi Setelah Tuning")
print("="*70)

eval_rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
eval_mae  = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mae")
eval_r2   = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")

preds_tuned = best_model.transform(test_df)
rmse_after  = eval_rmse.evaluate(preds_tuned)
mae_after   = eval_mae.evaluate(preds_tuned)
r2_after    = eval_r2.evaluate(preds_tuned)

# Ambil hasil RF sebelum tuning
rf_before_results = all_results.get("Random Forest", (None, None, None))
rmse_before, mae_before, r2_before = rf_before_results

print("\nPerbandingan Sebelum vs Sesudah Tuning (Random Forest):")
print(f"{'Metrik':<10} {'Sebelum':>12} {'Sesudah':>12} {'Perubahan':>12}")
print("-" * 48)

if rmse_before is not None:
    print(f"{'RMSE':<10} {rmse_before:>12.4f} {rmse_after:>12.4f} {rmse_after - rmse_before:>+12.4f}")
    print(f"{'MAE':<10} {mae_before:>12.4f} {mae_after:>12.4f}  {mae_after - mae_before:>+12.4f}")
    print(f"{'R²':<10} {r2_before:>12.4f} {r2_after:>12.4f}  {r2_after - r2_before:>+12.4f}")
else:
    print(f"{'RMSE':<10} {'N/A':>12} {rmse_after:>12.4f}")
    print(f"{'MAE':<10} {'N/A':>12} {mae_after:>12.4f}")
    print(f"{'R²':<10} {'N/A':>12} {r2_after:>12.4f}")

# ============================================================================
# 6. Feature Importance (Console Output Only)
# ============================================================================

print("\n" + "="*70)
print("Feature Importance")
print("="*70)

importances = best_model.featureImportances.toArray()

# Create tuples and sort by importance
feat_importances = list(zip(FEATURE_COLS, importances))
feat_importances.sort(key=lambda x: x[1], reverse=True)

print(f"\n{'Feature':<25} {'Importance':>12}")
print("-"*40)
for feature, importance in feat_importances:
    print(f"{feature:<25} {importance:>12.6f}")


# ============================================================================
# 8. Simpan Model Final
# ============================================================================

os.makedirs("../models", exist_ok=True)
best_model.write().overwrite().save("../models/final_model")
print("\n Model final disimpan di ../models/final_model")

# ============================================================================
# Finish
# ============================================================================

print("\n" + "="*70)
print(" HYPERPARAMETER TUNING SELESAI")
print("="*70)

spark.stop()
print("\n SparkSession ditutup.")
print(" Semua output tersimpan di folder outputs/")
print(" Model final tersimpan di folder models/")
