# Analisis Perbandingan Random Forest dan CNN pada Klasifikasi Citra Tutupan Lahan EuroSAT

## Deskripsi Proyek

Proyek ini bertujuan untuk membandingkan performa algoritma **Random Forest** dan **Convolutional Neural Network (CNN)** dalam mengklasifikasikan citra satelit berdasarkan kategori tutupan lahan menggunakan dataset EuroSAT.

Klasifikasi tutupan lahan merupakan salah satu penerapan *Machine Learning* dan *Deep Learning* dalam bidang penginderaan jauh (*remote sensing*). Informasi tutupan lahan dapat dimanfaatkan sebagai salah satu sumber data untuk pemantauan lingkungan dan perencanaan wilayah.

Eksperimen ini membandingkan kedua model berdasarkan accuracy, precision, recall, macro F1-score, dan waktu pelatihan.

## Tujuan

1. Mengimplementasikan Random Forest untuk klasifikasi citra tutupan lahan.
2. Mengimplementasikan CNN untuk mempelajari pola visual pada citra satelit.
3. Membandingkan performa kedua model menggunakan metrik evaluasi yang sama.
4. Menganalisis kelas tutupan lahan yang relatif mudah dan sulit diklasifikasikan.
5. Mendokumentasikan hasil eksperimen agar dapat direproduksi.

## Dataset

Dataset yang digunakan adalah **EuroSAT RGB**, yang berisi citra satelit Sentinel-2 dengan 10 kategori tutupan lahan.

- Sumber dataset: [EuroSAT RGB di Hugging Face](https://huggingface.co/datasets/giswqs/EuroSAT_RGB)
- Total data: 27.000 gambar.
- Jumlah kelas: 10.
- Format data yang digunakan: Parquet.
- Kolom dataset: `image`, `label`, dan `filename`.

### Kategori Tutupan Lahan

| No. | Kelas |
|---:|---|
| 1 | AnnualCrop |
| 2 | Forest |
| 3 | HerbaceousVegetation |
| 4 | Highway |
| 5 | Industrial |
| 6 | Pasture |
| 7 | PermanentCrop |
| 8 | Residential |
| 9 | River |
| 10 | SeaLake |

### Pembagian Dataset

Dataset memiliki tiga bagian:

| Bagian | Jumlah gambar |
|---|---:|
| Train | 18.900 |
| Validation | 5.400 |
| Test | 2.700 |
| **Total** | **27.000** |

Pada eksperimen awal, digunakan subset seimbang sebanyak 300 gambar per kelas untuk training, 100 gambar per kelas untuk validation, dan 100 gambar per kelas untuk testing.

Dengan demikian, eksperimen menggunakan total 5.000 gambar, terdiri atas 3.000 data training, 1.000 data validation, dan 1.000 data testing.

Seluruh gambar diubah ukurannya menjadi 32 × 32 piksel. Data citra dinormalisasi ke rentang 0–1 untuk CNN, sedangkan Random Forest menggunakan fitur piksel yang diratakan menjadi vektor.

## Metode

### 1. Random Forest

Random Forest merupakan algoritma *ensemble learning* yang menggabungkan prediksi dari sejumlah pohon keputusan.

Dalam proyek ini, citra diubah menjadi vektor fitur piksel sebelum digunakan untuk pelatihan.

Konfigurasi utama:
- Jumlah estimator: 200 pohon.
- Random state: 42.
- Pemrosesan paralel: diaktifkan menggunakan `n_jobs=-1`.

### 2. Convolutional Neural Network (CNN)

CNN merupakan jaringan saraf yang dirancang untuk mempelajari fitur spasial pada citra, seperti pola, tekstur, dan bentuk.

Arsitektur yang digunakan:
- Input citra RGB berukuran 32 × 32 piksel.
- Dua lapisan Conv2D.
- Dua lapisan MaxPooling2D.
- Lapisan Flatten.
- Dense layer dengan 64 neuron.
- Dropout sebesar 0,3.
- Output softmax dengan 10 kelas.

Konfigurasi pelatihan:
- Optimizer: Adam.
- Loss function: Sparse Categorical Crossentropy.
- Epoch: 5.
- Batch size: 32.

### Alur Eksperimen

1. Memuat dataset EuroSAT dari file Parquet.
2. Memilih subset data yang seimbang untuk setiap kelas.
3. Mengubah ukuran dan memproses citra.
4. Melatih model Random Forest.
5. Melatih model CNN dengan data validasi.
6. Mengevaluasi kedua model menggunakan test set yang sama.
7. Menyimpan hasil evaluasi dalam format CSV dan grafik.

## Hasil Eksperimen

Hasil berikut merupakan hasil aktual dari eksperimen awal yang dijalankan.

| Metrik | Random Forest | CNN |
|---|---:|---:|
| Accuracy | 0,6390 | 0,5110 |
| Macro Precision | 0,6273 | 0,5202 |
| Macro Recall | 0,6390 | 0,5110 |
| Macro F1-score | 0,6217 | 0,4929 |
| Waktu pelatihan | 3,87 detik | 8,98 detik |

### Analisis Hasil

Pada eksperimen awal, Random Forest memperoleh accuracy sebesar 63,9%, sedangkan CNN memperoleh accuracy sebesar 51,1%. Random Forest unggul sebesar 12,8 poin persentase pada dataset dan konfigurasi yang digunakan.

Random Forest juga menghasilkan macro F1-score sebesar 0,6217, lebih tinggi dibandingkan CNN sebesar 0,4929. Selain itu, waktu pelatihan Random Forest lebih singkat pada pengujian ini.

Berdasarkan classification report, kelas `SeaLake` memiliki F1-score tertinggi pada Random Forest, yaitu 0,85. Sementara itu, kelas `Highway` merupakan salah satu kelas yang paling sulit dikenali oleh kedua model, dengan F1-score 0,25 pada Random Forest dan 0,24 pada CNN.

Hasil tersebut menunjukkan bahwa performa model berbeda antar-kelas. Kesamaan karakteristik visual antara beberapa kategori dapat menjadi salah satu faktor yang memengaruhi kesalahan klasifikasi.

Namun, hasil ini belum membuktikan bahwa Random Forest secara umum selalu lebih unggul daripada CNN. CNN masih dapat ditingkatkan melalui penambahan data training, augmentasi citra, penyesuaian arsitektur, dan optimasi hyperparameter.

## Teknologi yang Digunakan

- Python
- TensorFlow / Keras
- Scikit-learn
- Pandas
- NumPy
- Pillow
- Matplotlib
- PyArrow
- Visual Studio Code
- Git dan GitHub

## Struktur Folder

```text
klasifikasi-tutupan-lahan/
├── data/
│   ├── train-00000-of-00001.parquet
│   ├── validation-00000-of-00001.parquet
│   └── test-00000-of-00001.parquet
├── results/
│   ├── perbandingan_model.csv
│   ├── confusion_matrix.png
│   └── perbandingan_grafik.png
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

File dataset tidak harus disertakan dalam repositori GitHub. Pengguna dapat mengunduhnya dari sumber resmi yang ditautkan di bagian Dataset.

## Instalasi dan Cara Menjalankan

### 1. Clone repositori

```bash
git clone URL_REPOSITORI_KAMU
cd klasifikasi-tutupan-lahan
```

Ganti `URL_REPOSITORI_KAMU` dengan URL repositori GitHub milikmu.

### 2. Buat virtual environment

Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Instal dependensi

```bash
python -m pip install -r requirements.txt
```

Pastikan file Parquet sudah diunduh dan ditempatkan di dalam folder `data/` sesuai nama file yang digunakan oleh `main.py`.

### 4. Jalankan program

```bash
python main.py
```

Program akan memuat subset dataset, melatih kedua model, menampilkan metrik evaluasi, dan menyimpan hasilnya ke folder `results/`.

## Output Program

Program menghasilkan tiga file utama:

- `perbandingan_model.csv`: tabel metrik evaluasi dan waktu pelatihan.
- `confusion_matrix.png`: visualisasi perbandingan hasil klasifikasi setiap kelas.
- `perbandingan_grafik.png`: grafik perbandingan accuracy dan macro F1-score.

## Keterbatasan

1. Eksperimen awal hanya menggunakan sebagian kecil dari keseluruhan dataset EuroSAT.
2. Resolusi citra diturunkan menjadi 32 × 32 piksel sehingga sebagian detail visual dapat hilang.
3. CNN dilatih selama lima epoch dengan arsitektur sederhana.
4. Hasil evaluasi dapat berubah akibat konfigurasi model, pembagian data, dan kondisi perangkat.
5. Dataset ini digunakan untuk klasifikasi kategori tutupan lahan, bukan untuk mendeteksi perubahan tutupan hutan dari waktu ke waktu atau membuktikan terjadinya deforestasi.

## Rencana Pengembangan

- Meningkatkan jumlah data training.
- Menambahkan augmentasi citra pada CNN.
- Menerapkan EarlyStopping dan menyimpan model terbaik berdasarkan validation loss.
- Melakukan tuning hyperparameter.
- Membandingkan hasil dari beberapa random seed.
- Menguji model menggunakan seluruh test set setelah konfigurasi ditetapkan.
- Menyimpan model terlatih agar dapat digunakan kembali untuk prediksi citra baru.

## Kesimpulan

Eksperimen awal menunjukkan bahwa Random Forest menghasilkan performa lebih baik dibandingkan CNN berdasarkan accuracy, macro precision, macro recall, macro F1-score, dan waktu pelatihan pada konfigurasi yang diuji.

Hasil ini menjadi dasar untuk eksperimen lanjutan guna mengetahui pengaruh jumlah data, preprocessing, dan arsitektur model terhadap performa klasifikasi citra tutupan lahan.

---

**Catatan:** Hasil yang dicantumkan merupakan hasil eksperimen awal, bukan hasil akhir dari seluruh kemungkinan konfigurasi model.
