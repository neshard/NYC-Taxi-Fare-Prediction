# Panduan Menjalankan Python Scripts (Tanpa Jupyter Notebook)

Anda sekarang memiliki tiga file Python untuk menghindari timeout di Jupyter:

- **04_modeling.py** - Feature engineering & training model
- **05_evaluation.py** - Evaluasi dan perbandingan model
- **06_tuning.py** - Hyperparameter tuning

## Keuntungan Menggunakan File Python vs Notebook

✅ **Tidak ada timeout** (berbeda dengan Jupyter local mode)
✅ **Lebih cepat** pada data besar
✅ **Cocok untuk production**
✅ **Mudah di-schedule** dengan cron/task scheduler

## Cara Menjalankan

### Opsi 1: Jalankan Semua Sekaligus (Rekomendasi)

```bash
python run_pipeline.py
```

Output:
```
======================================================================
                    NYC Taxi Fair Prediction Pipeline
======================================================================

Scripts yang akan dijalankan:
  1. 04_modeling.py
  2. 05_evaluation.py
  3. 06_tuning.py

======================================================================
```

### Opsi 2: Jalankan Individual

#### Step 1 - Modeling
```bash
python 04_modeling.py
```

Ini akan:
- ✓ Load data dari `data/raw/nyc_taxi_data.csv`
- ✓ Feature engineering (haversine distance, time features)
- ✓ Split data 80/20
- ✓ Training 3 model: Linear Regression, Random Forest, GBT
- ✓ Simpan trained models ke `models/`

#### Step 2 - Evaluation
```bash
python 05_evaluation.py
```

Ini akan:
- ✓ Load model dari `models/`
- ✓ Evaluasi dengan metrik: RMSE, MAE, R²
- ✓ Generate grafik perbandingan model
- ✓ Simpan hasil di `outputs/evaluation_results.csv`

#### Step 3 - Tuning
```bash
python 06_tuning.py
```

Ini akan:
- ✓ Load best model dari evaluasi
- ✓ Hyperparameter tuning dengan CrossValidator
- ✓ Bandingkan sebelum vs sesudah tuning
- ✓ Tampilkan feature importance
- ✓ Simpan final model ke `models/final_model`

## Struktur Output

Setelah selesai, akan ada file-file baru:

```
outputs/
├── evaluation_results.csv      # Hasil evaluasi model (RMSE, MAE, R²)
├── model_comparison.png        # Grafik perbandingan 3 model
├── pred_vs_actual.png          # Scatter plot prediksi vs aktual
├── tuning_comparison.png       # Grafik sebelum vs sesudah tuning
└── feature_importance.png      # Ranking feature importance

models/
├── lr_model/                   # Linear Regression model
├── rf_model/                   # Random Forest model
├── gbt_model/                  # Gradient Boosted Trees model
└── final_model/                # Best model after tuning

data/
└── test_data.parquet           # Test set untuk evaluation
```

## Troubleshooting

### Error: "No module named 'pyspark'"

Install PySpark:
```bash
pip install pyspark
```

### Error: "file not found"

Pastikan:
1. Running dari folder project root (di mana file Python berada)
2. Data file ada di `data/raw/nyc_taxi_data.csv` atau `train.csv`
3. Folder `models/` dan `outputs/` akan dibuat otomatis

### Error: "TimeoutError" atau "Worker timeout"

**Ini sudah solved!** Dengan menggunakan Python script bukan Jupyter notebook.

Jika masih timeout:
- Coba reduce `SAMPLE_SIZE` di `04_modeling.py` (line ~150)
- Atau jalankan di cloud (Databricks/AWS EMR)

### Script tidak jalan di Windows PowerShell

Jalankan dengan explicit python:
```powershell
python 04_modeling.py
```

Atau gunakan Command Prompt biasa:
```cmd
python 04_modeling.py
```

## Konfigurasi Advanced

### Mengubah Sample Size (untuk testing cepat)

Di `04_modeling.py`, cari section "Persiapan Fitur & Split Data":

```python
SAMPLE_SIZE = 10000  # Default: 10K untuk test cepat
# SAMPLE_SIZE = None  # Uncomment untuk gunakan semua data
# SAMPLE_SIZE = 25000  # Atau uncomment untuk 25K
```

### Mengubah Model Parameters

Di `04_modeling.py`, section "Training Model":

```python
# Linear Regression
lr = LinearRegression(
    featuresCol="features", 
    labelCol="label", 
    maxIter=20,         # Ubah ini
    regParam=0.01       # Atau ini
)

# Random Forest
rf = RandomForestRegressor(
    featuresCol="features", 
    labelCol="label", 
    numTrees=50,        # Ubah jumlah tree
    seed=42, 
    maxDepth=10         # Ubah kedalaman
)
```

## Next Steps

Setelah selesai:

1. **Review hasil** di folder `outputs/`
   - Lihat `evaluation_results.csv` untuk performa model
   - Lihat grafik untuk visualisasi

2. **Gunakan final model** untuk prediksi
   ```python
   from pyspark.ml.regression import RandomForestRegressionModel
   
   model = RandomForestRegressionModel.load("models/final_model")
   predictions = model.transform(new_data)
   ```

3. **Deploy ke production**
   - Export model ke format lain (ONNX, PMML)
   - Atau deploy langsung dengan PySpark

## Tips & Best Practices

⭐ **Start with sample data** (10K rows) untuk testing cepat
⭐ **Increase data size** setelah code berjalan
⭐ **Monitor memory** saat training data besar
⭐ **Save intermediate results** untuk debugging
⭐ **Version control** hasil model dengan DVC atau MLflow

---

**Happy modeling! 🚕📊**

Untuk pertanyaan, lihat docstring di masing-masing file Python.
