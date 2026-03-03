# DATA UNDERSTANDING

## Studi Kasus: Iris Flower Dataset

---

## 2.1 Pendahuluan

Dalam metodologi **CRISP-DM (Cross Industry Standard Process for Data Mining)**, tahap *Data Understanding* dilakukan setelah tahap *Business Understanding*. Tahap ini bertujuan untuk memahami dataset yang akan digunakan sebelum memasuki proses *Data Preparation* dan *Modeling*. 

Data Understanding sangat penting karena menjadi fondasi dalam analisis data. Pemahaman yang kurang terhadap struktur, tipe data, distribusi, serta kualitas data dapat menyebabkan kesalahan dalam pemodelan dan menghasilkan kesimpulan yang bias atau tidak valid.

Pada studi kasus ini digunakan **Iris Flower Dataset**, yaitu dataset publik yang sering digunakan dalam pembelajaran data mining dan machine learning, khususnya untuk kasus klasifikasi.

---

## 2.2 Tujuan Data Understanding

Tujuan dari tahap ini adalah:

1. Memastikan dataset dapat dibaca dengan benar.
2. Memahami struktur dan tipe data setiap atribut.
3. Mengidentifikasi adanya missing value atau duplikasi data.
4. Mengetahui karakteristik statistik data numerik.
5. Melihat distribusi kelas (target).

---

## 2.3 Sumber dan Karakteristik Dataset

Dataset yang digunakan adalah **Iris Flower Dataset** yang memiliki karakteristik sebagai berikut:

- Jumlah data: **150 baris**
- Jumlah atribut: **5 kolom**
- 4 atribut numerik:
  - Sepal Length
  - Sepal Width
  - Petal Length
  - Petal Width
- 1 atribut kategorikal (target):
  - Species (Setosa, Versicolor, Virginica)

Dataset ini digunakan untuk membangun model klasifikasi jenis bunga berdasarkan ukuran sepal dan petal.

---

## 2.4 Memuat Dataset

Dataset dibaca menggunakan fungsi `read_csv()` dari library **pandas**.  
File `IRIS.csv` harus berada dalam folder yang sama dengan notebook atau project Jupyter Book agar dapat diakses tanpa error.

Langkah ini penting untuk memastikan bahwa dataset berhasil dimuat sebelum dilakukan analisis lebih lanjut.

```python
import pandas as pd

df = pd.read_csv("IRIS.csv")
df.head()

## 2.5 Dimensi Dataset

Untuk mengetahui ukuran dataset (jumlah baris dan kolom) digunakan:

```python
df.shape
```

### Interpretasi

Output berbentuk `(baris, kolom)`.

Pada Iris dataset umumnya akan muncul `(150, 5)`, artinya:

- 150 data bunga
- 5 kolom (4 fitur + 1 target)

Informasi dimensi ini penting untuk:

- Memastikan data tidak terpotong saat dibaca
- Mengetahui skala dataset sebelum visualisasi atau pemodelan

---

## 2.6 Informasi Dataset (Tipe Data & Non-Null)

Untuk memeriksa tipe data setiap kolom dan jumlah nilai tidak kosong digunakan:

```python
df.info()
```

### Interpretasi

- 4 fitur numerik biasanya bertipe `float64`
- 1 kolom target `species` bertipe `object` atau `category`
- Bagian *Non-Null Count* membantu mendeteksi missing value secara cepat  
  (jika jumlah non-null < 150 berarti ada data kosong)

### Mengapa ini penting?

Tipe data menentukan proses preprocessing:

- Data numerik bisa dilakukan scaling atau normalisasi
- Data kategorikal perlu encoding jika digunakan dalam model

Kesalahan tipe data (misalnya angka terbaca sebagai teks) dapat mengganggu analisis.

---

## 2.7 Statistik Deskriptif

Ringkasan statistik fitur numerik dapat dilihat dengan:

```python
df.describe()
```

### Penjelasan Output

- `count` : jumlah data valid (non-null)
- `mean` : rata-rata
- `std` : standar deviasi (tingkat variasi)
- `min/max` : nilai minimum dan maksimum
- `25%/50%/75%` : kuartil (Q1, median, Q3)

### Interpretasi Umum pada Iris Dataset

- Fitur petal biasanya memiliki variasi yang lebih “tajam” dibanding sepal.
- Standar deviasi (`std`) pada fitur petal sering menunjukkan pemisahan antar kelas.
- Rentang nilai (min–max) membantu memahami skala tiap fitur.

---

## 2.8 Pengecekan Missing Value

Untuk memastikan tidak ada nilai kosong:

```python
df.isnull().sum()
```

### Interpretasi

- Jika semua kolom bernilai 0, berarti tidak ada missing value.
- Jika ada kolom bernilai > 0, maka perlu penanganan pada tahap Data Preparation seperti:
  - imputasi (mean/median/modus)
  - atau menghapus baris tertentu (jika jumlahnya sedikit)

Pada dataset Iris standar, umumnya tidak terdapat missing value.

---

## 2.9 Pengecekan Duplikasi

Untuk memastikan tidak ada baris data ganda:

```python
df.duplicated().sum()
```

### Interpretasi

- Jika output `0`, berarti tidak ada data duplikat.
- Jika output > 0, maka duplikasi dapat mempengaruhi model karena:
  - Dataset menjadi bias pada pola tertentu
  - Evaluasi model bisa tampak lebih bagus karena data yang sama muncul berulang

Jika ditemukan duplikasi, penanganan umumnya:

```python
df = df.drop_duplicates()
```

---

## 2.10 Distribusi Kelas (Target)

Untuk mengetahui jumlah data tiap kelas:

```python
df['species'].value_counts()
```

### Interpretasi

Pada Iris dataset, umumnya masing-masing kelas berjumlah 50:

- Setosa: 50
- Versicolor: 50
- Virginica: 50

Jika jumlahnya seimbang, dataset termasuk **balanced dataset**.  
Hal ini menguntungkan karena model tidak condong ke salah satu kelas.

---

## 2.11 Visualisasi Distribusi Kelas

Untuk menampilkan distribusi kelas secara visual:

```python
sns.countplot(x='species', data=df)
plt.title("Distribusi Jumlah Data per Species")
plt.show()
```

### Penjelasan

- Grafik batang menunjukkan jumlah data setiap species.
- Jika tinggi batang sama atau hampir sama, maka kelas seimbang.
- Jika ada batang jauh lebih tinggi, maka dataset imbalanced dan perlu penanganan (misalnya oversampling atau undersampling).

---

## 2.12 Histogram Fitur Numerik

Histogram digunakan untuk melihat sebaran nilai tiap fitur:

```python
df.hist(figsize=(10,8))
plt.show()
```

### Interpretasi Umum

Histogram membantu memahami apakah data:

- Menyebar normal
- Condong ke kiri atau kanan (skew)
- Memiliki beberapa kelompok nilai (multi-modal)

Pada Iris dataset, fitur petal biasanya menunjukkan pola yang lebih terpisah dibanding fitur sepal, karena Setosa memiliki ukuran petal yang jauh lebih kecil.

---

## 2.13 Boxplot untuk Deteksi Outlier

Boxplot membantu melihat sebaran data dan kemungkinan outlier:

```python
plt.figure(figsize=(10,6))
sns.boxplot(data=df[['sepal_length','sepal_width','petal_length','petal_width']])
plt.title("Boxplot Fitur Numerik Iris Dataset")
plt.show()
```

### Penjelasan

- Garis tengah box = median
- Box = rentang kuartil (Q1–Q3)
- Whisker = rentang nilai wajar
- Titik di luar whisker = kemungkinan outlier

### Interpretasi Umum

Iris dataset biasanya tidak memiliki outlier ekstrem yang mengganggu.  
Jika ada outlier, perlu dianalisis apakah merupakan kesalahan input atau variasi alami.

---

## 2.14 Analisis Fitur Petal

Bagian ini fokus pada dua fitur paling informatif:

- `petal_length`
- `petal_width`

### a) Statistik Deskriptif Petal

```python
df[['petal_length', 'petal_width']].describe()
```

#### Interpretasi

- Petal memiliki rentang nilai yang lebih kontras antar kelas.
- Setosa memiliki petal kecil.
- Versicolor dan Virginica memiliki petal lebih besar.
- Karena perbedaannya jelas, fitur petal sering menjadi fitur utama dalam klasifikasi Iris.

---

### b) Histogram Fitur Petal

```python
df[['petal_length', 'petal_width']].hist(figsize=(8,6))
plt.suptitle("Histogram Petal Length dan Petal Width")
plt.show()
```

Histogram memperlihatkan pola sebaran khusus petal dan biasanya menunjukkan pemisahan yang kuat antara Setosa dan dua kelas lainnya.

---

### c) Boxplot Petal

```python
sns.boxplot(data=df[['petal_length', 'petal_width']])
plt.title("Boxplot Petal Length dan Petal Width")
plt.show()
```

Boxplot menunjukkan bahwa petal Setosa lebih kecil secara konsisten dan tidak terdapat outlier signifikan.

---

### d) Scatter Plot Petal Length vs Petal Width

```python
sns.scatterplot(x='petal_length', y='petal_width', hue='species', data=df)
plt.title("Petal Length vs Petal Width")
plt.show()
```

#### Interpretasi

- Setosa terpisah sangat jelas (cluster sendiri).
- Versicolor dan Virginica sedikit overlap.
- Hubungan petal_length dan petal_width bersifat positif (semakin panjang, semakin lebar).

Kombinasi fitur petal sangat efektif dalam membedakan kelas.

---

## 2.15 Kesimpulan Tahap Data Understanding

Berdasarkan eksplorasi yang telah dilakukan:

- Dataset berhasil dibaca dengan baik (umumnya 150×5).
- Tipe data sesuai: fitur numerik dan target kategorikal.
- Tidak ditemukan missing value.
- Tidak ditemukan data duplikat.
- Distribusi kelas seimbang (balanced dataset).
- Fitur petal memberikan pemisahan kelas paling jelas.

Dataset siap untuk masuk ke tahap berikutnya yaitu **Data Preparation** sebelum pemodelan dilakukan.