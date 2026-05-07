#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║  📈 NOTEBOOK 2: Model Evaluation & Performance Comparison                 ║
║     NYC Taxi Fare Prediction — Hanif's Work                               ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
  This script evaluates and compares the three baseline models trained in
  the previous notebook, identifying the best-performing algorithm:
  - Load trained models from disk (checkpoint files)
  - Generate predictions on hold-out test set (20% unseen data)
  - Calculate regression performance metrics
  - Rank models by accuracy and generalization capability
  - Save benchmark results for reporting

INPUT ARTIFACTS:
  Models Loaded From:
  - models/lr_model/       (Linear Regression)
  - models/rf_model/       (Random Forest)
  - models/gbt_model/      (Gradient Boosted Trees)
  
  Test Data:
  - Reconstructed from original data (same split as 04_modeling.py)
  - 20% hold-out set with fixed seed (reproducible)
  - Completely unseen by training algorithms

OUTPUT ARTIFACTS:
  - outputs/evaluation_results.parquet (benchmark metrics table)
  - Console output: Model rankings by RMSE/MAE/R²

KEY OPERATIONS:
  1. Model Loading
     └─ Deserialize trained ML models from Parquet format
  
  2. Test Data Reconstruction
     ├─ Reload raw data with identical preprocessing
     ├─ Apply same feature engineering pipeline
     └─ Use same random seed for reproducibility
  
  3. Predictions Generation
     └─ Each model predicts fare_amount on test set
  
  4. Performance Metrics Calculation
     ├─ RMSE (Root Mean Squared Error)
     │  └─ Dollar amount of typical prediction error (penalizes outliers)
     ├─ MAE (Mean Absolute Error)  
     │  └─ Median deviation (robust to extreme errors)
     └─ R² (Coefficient of Determination)
        └─ % of fare variance explained by the model (0-1 scale)
  
  5. Model Ranking
     ├─ Sort by RMSE (lower is better)
     └─ Identify best model for hyperparameter tuning
  
  6. Results Storage
     └─ Save benchmark table to parquet for downstream analysis

METRIC INTERPRETATION:
  RMSE = $4.82
    → Average prediction error magnitude is $4.82
    → For a predicted $20 fare: actual is likely $15.18-$24.82
  
  MAE = $2.36
    → Median absolute deviation from actual fare
    → More robust than RMSE (doesn't penalize outliers as heavily)
  
  R² = 0.7677
    → Model explains 76.77% of fare variance
    → 23.23% of variance due to unmeasured factors (surge pricing, events, etc)
    → R² > 0.70 is considered "good" for real-world regression problems

MODEL COMPARISON SAMPLE:
  ┌─────────────────────┬────────┬────────┬────────┐
  │ Model               │ RMSE   │ MAE    │ R²     │
  ├─────────────────────┼────────┼────────┼────────┤
  │ Random Forest       │ 4.8163 │ 2.3607 │ 0.7677 │ ← BEST
  │ Gradient Boosting   │ 5.0351 │ 2.3818 │ 0.7462 │
  │ Linear Regression   │ 9.8438 │ 5.9664 │ 0.0297 │ ← WORST
  └─────────────────────┴────────┴────────┴────────┘

KEY FINDING:
  Random Forest is 25+ percentage points better than Linear Regression (R²),
  proving that non-linear tree-based methods are superior for this problem.
  This validates the shift from v1 (Random Forest) to v2+ optimization phases.

PREREQUISITE:
  04_modeling.py must be run first to generate model checkpoints.

NEXT STEP:
  Run 06_tuning.py to optimize the best model via hyperparameter tuning.
"""

# ============================================================================
# 1. Setup
# ============================================================================

# PENTING: Set JAVA_TOOL_OPTIONS SEBELUM import PySpark agar JVM subprocess
# mewarisi flag ini saat diluncurkan oleh py4j (fix untuk Java 17+/21+/23+)
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

from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegressionModel, RandomForestRegressionModel, GBTRegressionModel
from pyspark.ml.evaluation import RegressionEvaluator

spark = SparkSession.builder \
    .appName("NYC_Taxi_Task5_Evaluation") \
    .config("spark.driver.memory", "4g") \
    .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
print("SparkSession aktif:", spark.version)

# ============================================================================
# 2. Load Model & Test Data
# ============================================================================

from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler

print("Merekonstruksi test set dari data raw (karena parquet tidak disimpan)...")
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

_, test_df = df_model.randomSplit([0.8, 0.2], seed=42)
print(f"Test set direkonstruksi: {test_df.count():,} baris")

# Load semua model
try:
    lr_model  = LinearRegressionModel.load("../models/lr_model")
    rf_model  = RandomForestRegressionModel.load("../models/rf_model")
    gbt_model = GBTRegressionModel.load("../models/gbt_model")
    print("✅ Semua model berhasil di-load.")
except Exception as e:
    print(f"⚠️  Error loading models: {e}")
    print("💡 Pastikan 04_modeling.py telah dijalankan untuk menghasilkan model")
    print("   atau model sudah tersimpan di folder models/")
    spark.stop()
    exit(1)

# ============================================================================
# 3. Evaluasi — RMSE, MAE, R²
# ============================================================================

evaluator_rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
evaluator_mae  = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mae")
evaluator_r2   = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")

def evaluate_model(model, test_data, model_name):
    preds = model.transform(test_data)
    rmse  = evaluator_rmse.evaluate(preds)
    mae   = evaluator_mae.evaluate(preds)
    r2    = evaluator_r2.evaluate(preds)
    print(f"\n{'='*45}")
    print(f" Model : {model_name}")
    print(f" RMSE  : {rmse:.4f}")
    print(f" MAE   : {mae:.4f}")
    print(f" R²    : {r2:.4f}")
    return (model_name, rmse, mae, r2)

results = []
results.append(evaluate_model(lr_model,  test_df, "Linear Regression"))
results.append(evaluate_model(rf_model,  test_df, "Random Forest"))
results.append(evaluate_model(gbt_model, test_df, "Gradient Boosted Trees"))

# ============================================================================
# 4. Tabel Perbandingan
# ============================================================================

print("\n" + "="*70)
print("RINGKASAN EVALUASI (urut dari terbaik)")
print("="*70)
print(f"{'Model':<25} {'RMSE':>12} {'MAE':>12} {'R²':>12}")
print("-"*70)

# Sort by RMSE (ascending - lebih kecil lebih baik)
sorted_results = sorted(results, key=lambda x: x[1])
best_model_name = sorted_results[0][0]

for model_name, rmse, mae, r2 in sorted_results:
    print(f"{model_name:<25} {rmse:>12.4f} {mae:>12.4f} {r2:>12.4f}")

# Simpan hasil ke parquet untuk referensi
os.makedirs("../outputs", exist_ok=True)

# Simpan hasil ke parquet untuk referensi menggunakan pandas
# (Menghindari py4j Python worker crash saat spark.createDataFrame)
import pandas as pd

df_results = pd.DataFrame(results, columns=["Model", "RMSE", "MAE", "R2"])
df_results.to_parquet("../outputs/evaluation_results.parquet", index=False)
print("\n✓ Hasil disimpan di ../outputs/evaluation_results.parquet")
print(f"✓ Model terbaik: {best_model_name}")

# ============================================================================
# 5. Interpretasi Hasil Evaluasi (Untuk Laporan Tugas)
# ============================================================================

print("\n" + "="*70)
print("INTERPRETASI HASIL EVALUASI UNTUK LAPORAN TUGAS")
print("="*70)
print("1. Apakah model ini sudah baik?")
print("   - Ya, untuk sebuah 'baseline model' (model awal sebelum tuning),")
print("     hasil Random Forest sudah terbilang cukup baik.")
print("   - R-Squared (R2) ~ 0.76 artinya model berhasil memahami dan menjelaskan")
print("     sekitar 76% faktor yang mempengaruhi fluktuasi harga taksi.")
print("   - RMSE ~ 4.81 artinya rata-rata tebakan model kita meleset sekitar $4.81.")
print("     Mengingat tarif taksi di NYC sangat fluktuatif, angka ini sangat wajar.")
print("")
print("2. Mengapa Linear Regression sangat buruk (R2 ~ 0.02)?")
print("   - Algoritma regresi linear sederhana kurang mampu menangkap pola non-linear")
print("     di dunia nyata (seperti perhitungan jarak dari koordinat bumi).")
print("   - Sebaliknya, Random Forest jauh lebih cerdas melihat pola kompleks tersebut.")
print("="*70)
# ============================================================================
# 5. Tampilkan Sampel Prediksi (Actual vs Predicted)
# ============================================================================

print("\n" + "="*70)
print("SAMPEL PREDIKSI (Model Terbaik: Random Forest)")
print("="*70)

# Gunakan model terbaik (Random Forest) untuk memprediksi test set
predictions = rf_model.transform(test_df)

# Tampilkan 15 data pertama untuk membandingkan Actual Fare vs Predicted Fare
predictions.select("label", "prediction") \
    .withColumnRenamed("label", "Actual_Fare_($)") \
    .withColumnRenamed("prediction", "Predicted_Fare_($)") \
    .show(15, truncate=False)

# ============================================================================
# Finish
# ============================================================================

spark.stop()
print("\n✅ Evaluasi selesai.")
print(f"Lanjut ke 06_tuning.py untuk hyperparameter tuning.")
