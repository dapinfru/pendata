# Data Understanding

Data Understanding adalah tahap dalam metodologi CRISP-DM yang dilakukan setelah memahami tujuan bisnis. Pada tahap ini, kita mulai fokus pada data yang akan digunakan. Tujuannya adalah untuk benar-benar mengenal isi data sebelum masuk ke proses pengolahan atau pembuatan model.

Pada tahap ini dilakukan pemeriksaan terhadap jumlah data, jenis atribut, tipe data (numerik atau kategori), serta melihat apakah terdapat data yang kosong atau tidak sesuai. Dengan memahami data sejak awal, kita bisa menghindari kesalahan dalam analisis dan bisa menentukan metode yang tepat untuk digunakan.

Dalam dataset IRIS, terdapat 150 data bunga dengan 4 atribut numerik, yaitu panjang sepal, lebar sepal, panjang petal, dan lebar petal. Selain itu, terdapat 1 atribut kategori yaitu species (jenis bunga) yang terdiri dari Setosa, Versicolor, dan Virginica. Data ini digunakan untuk membangun model klasifikasi.

## 1.Sumber Data

Data yang digunakan dalam penelitian ini adalah dataset IRIS. Dataset ini merupakan dataset publik yang sering digunakan dalam pembelajaran data mining dan machine learning.

## 2. Eksplorasi Dataset

Pada tahap Data Understanding, eksplorasi data dilakukan untuk memahami isi, struktur, dan karakteristik dataset sebelum masuk ke tahap pengolahan atau pemodelan. Berikut adalah tahapan eksplorasi yang dilakukan menggunakan Python.

1. Import Library Yg di butuhkan 

Langkah pertama adalah mengimpor library yang akan digunakan untuk membaca dan menganalisis data.
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

```
Penjelasan:

- pandas → untuk membaca dan mengelola data

- numpy → untuk operasi numerik

- matplotlib dan seaborn → untuk visualisasi data
---------------------

2. Membaca Dataset
Dataset dibaca menggunakan pandas.
```
df = pd.read_csv("IRIS.csv")
df.head()

```
Tahap ini bertujuan untuk:

- Melihat 5 data pertama

- Memastikan dataset berhasil dibaca

- Mengetahui struktur awal data
----
3. Melihat Informasi Umum Dataset 
Untuk mengetahui jumlah data, tipe data, dan apakah ada data kosong:
```
df.info()

```
Dari sini bisa tau:

- Jumlah total baris dan kolom

- Tipe data setiap atribut

- Apakah ada nilai yang hilang (missing value)
----
4. Mengecek Missing Value
```
df.isnull().sum()

```
Tujuan:

- Memastikan apakah ada data kosong

- Jika ada, nanti perlu ditangani di tahap Data Preparation

Pada dataset IRIS biasanya tidak terdapat missing value.

---

5. Statistik Deskriptif
Untuk melihat ringkasan statistik dari data numerik:
```
df.describe()

```
Hasilnya akan menampilkan:

- Nilai minimum

- Nilai maksimum

- Rata-rata (mean)

- Standar deviasi

- Kuartil
---
6. Melihat Distribusi Kelas (Target)
```
df['species'].value_counts()

```
Tujuannya:

- Mengetahui apakah data seimbang atau tidak

- Melihat jumlah masing-masing kelas

Pada dataset IRIS, biasanya setiap kelas memiliki 50 data (balanced dataset).

--- 