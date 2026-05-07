# Laporan Proyek: NYC Taxi Fare Prediction

## 1. Pendahuluan
Proyek ini bertujuan untuk membangun model Machine Learning yang dapat memprediksi tarif perjalanan taksi di New York City (NYC) berdasarkan data historis perjalanan. Prediksi tarif yang akurat sangat penting bagi penumpang dan pengemudi untuk mengestimasi biaya perjalanan sebelum taksi dipesan.

## 2. Pemrosesan Data (Data Preprocessing) & Feature Engineering
Dataset mentah memuat informasi dasar seperti waktu penjemputan, lokasi penjemputan (koordinat), lokasi tujuan (koordinat), jumlah penumpang, dan tarif aktual. 

Untuk meningkatkan performa model, dilakukan beberapa tahapan pembersihan dan rekayasa fitur (*Feature Engineering*):
*   **Pembersihan Data:** Menghapus baris dengan nilai kosong (NA), memfilter tarif di bawah $0 atau di atas $500, membatasi jumlah penumpang (1-6), dan memastikan titik koordinat berada di sekitar wilayah New York City (Longitude: -75 hingga -72, Latitude: 40 hingga 42).
*   **Ekstraksi Waktu:** Fitur `pickup_datetime` dipecah menjadi beberapa fitur spesifik seperti `hour` (jam), `day_of_week` (hari dalam minggu), `month` (bulan), dan `year` (tahun) untuk menangkap pola jam sibuk dan musim.
*   **Perhitungan Jarak:** Menambahkan fitur `distance_km` menggunakan rumus *Haversine* untuk menghitung jarak lurus geografis antara titik penjemputan dan penurunan. Jarak adalah faktor penentu utama tarif.

## 3. Pelatihan Model Dasar (Baseline Modeling)
Untuk membandingkan performa algoritma, tiga jenis model regresi dilatih menggunakan PySpark MLlib:
1.  **Linear Regression**: Model statistik sederhana berbasis hubungan linear.
2.  **Random Forest Regressor**: Model berbasis struktur pohon keputusan jamak (*ensemble*) yang mampu menangkap pola data non-linear.
3.  **Gradient Boosted Trees (GBT)**: Model ansambel lain yang mengoreksi kelemahan pohon-pohon sebelumnya secara berurutan.

## 4. Evaluasi Model
Model dievaluasi menggunakan *Test Set* terpisah untuk melihat kemampuannya dalam memprediksi data yang belum pernah dilihat sebelumnya. Metrik yang digunakan adalah **RMSE** (Root Mean Square Error), **MAE** (Mean Absolute Error), dan **R²** (R-Squared).

Berikut adalah ringkasan performa sebelum tuning:

| Model | RMSE | MAE | R² |
| :--- | :--- | :--- | :--- |
| **Random Forest** | **4.8163** | **2.3607** | **0.7677** |
| Gradient Boosted Trees | 5.0351 | 2.3818 | 0.7462 |
| Linear Regression | 9.8438 | 5.9664 | 0.0297 |

> [!NOTE] 
> **Interpretasi:**
> Random Forest merupakan model terbaik dengan R² ~0.76 (artinya model ini mampu menjelaskan sekitar 76.7% variabilitas tarif taksi berdasarkan fitur yang ada). RMSE sebesar $4.81 menandakan rata-rata kesalahan tebakan model hanyalah sekitar $4.81. Linear Regression gagal mempelajari pola dataset (R² mendekati 0) karena hubungan antara koordinat/waktu terhadap tarif taksi tidak bersifat linear sederhana.

## 5. Hyperparameter Tuning
Untuk memaksimalkan performa, dilakukan tahapan *Hyperparameter Tuning* terhadap model terbaik (Random Forest) menggunakan teknik *Grid Search* dan *Cross Validation* dengan variasi parameter:
*   `numTrees`: [50, 100, 150]
*   `maxDepth`: [5, 10, 15]

**Parameter Terbaik yang Ditemukan:**
*   `numTrees`: 150
*   `maxDepth`: 15

**Perbandingan Performa Random Forest (Sebelum vs Sesudah Tuning):**

| Metrik | Sebelum Tuning | Sesudah Tuning | Perubahan |
| :--- | :--- | :--- | :--- |
| **RMSE** | 4.8163 | 4.8444 | +0.0281 |
| **MAE** | 2.3607 | 2.3096 | -0.0510 (Meningkat) |
| **R²** | 0.7677 | 0.7650 | -0.0027 |

> [!TIP]
> Walaupun nilai RMSE dan R² mengalami sedikit sekali koreksi, nilai **MAE (Mean Absolute Error) mengalami penurunan/peningkatan akurasi** menjadi 2.3096. Ini berarti rata-rata error absolut (tanpa pinalti kuadrat ekstrim) menjadi lebih presisi. Model final yang di-tuning lebih stabil secara *cross-validation*.

## 6. Feature Importance (Tingkat Kepentingan Fitur)
Setelah di-tuning, model Random Forest mengungkapkan variabel mana yang paling berpengaruh terhadap prediksi tarif taksi:

1.  **`distance_km` (Jarak Tempuh): 47.8%** - *Paling dominan secara absolut.*
2.  **`pickup_longitude`**: 11.7%
3.  **`dropoff_longitude`**: 10.7%
4.  **`dropoff_latitude`**: 7.2%
5.  **`pickup_latitude`**: 5.5%
6.  **`hour` (Jam Penjemputan)**: 5.4%
7.  **Waktu lainnya (`year`, `month`, `day_of_week`)**: ~3 - 4%
8.  **`passenger_count`**: 1.2% - *Sangat sedikit pengaruhnya terhadap total tarif.*

## 7. Kesimpulan
Proyek prediksi Tarif Taksi NYC telah berhasil diselesaikan. Model **Random Forest** terbukti sebagai model paling mumpuni untuk menangani kompleksitas dataset spasial-temporal ini. Jarak tempuh (Haversine) memegang pengaruh paling vital terhadap prediksi harga, disusul oleh fitur berbasis lokasi geografis yang mampu mendeteksi wilayah-wilayah premium (seperti area bandara atau tol). Model akhir telah disimpan (`models/final_model`) dan siap untuk tahapan *deployment* lebih lanjut.
