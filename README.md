# Analisis Bias pada Ulasan Aplikasi MyTelkomsel (Kelompok 1)

Repository ini berisi eksperimen dan pipeline pemrosesan bahasa alami (NLP) untuk menganalisis sentimen serta bias anotasi pada ulasan pengguna aplikasi **MyTelkomsel** di Google Play Store.

---

## 📁 Struktur Direktori Proyek

```text
Sentiment-Analysis-bias-mytelkomsel/
│
├── data/
│   ├── raw/                       # Data mentah (unduhan Play Store / Google Sheets)
│   └── processed/                 # Data hasil preprocessing (tokenized, normalized, clean)
│
├── notebooks/
│   └── DataPreprocessing_mytelkom.ipynb  # Jupyter Notebook utama (eksperimen & analisis)
│
├── src/                           # Modul kode Python yang terstruktur & reusable
│   ├── scraping/                  # Modul pengumpulan data
│   │   ├── playstore_scraper.py   # Scraping ulasan langsung via google-play-scraper
│   │   └── sheet_loader.py        # Pengambilan ulasan berlabel dari Google Sheets
│   ├── preprocessing/             # Pipeline preprocessing teks bahasa Indonesia
│   │   └── pipeline.py            # Tokenisasi, Normalisasi, Stopwords, & Stemming
│   └── evaluation/                # Evaluasi model & analisis bias
│       ├── bias_metrics.py        # Pengukuran kesepakatan anotator (Fleiss' Kappa)
│       └── model_training.py      # TF-IDF, CatBoost Classifier, LogReg, & 5-Fold CV
│
├── requirements.txt               # Daftar dependensi library Python
├── requirement.txt                # Alias dependensi (kompatibilitas)
├── .gitignore                     # Mengabaikan cache, checkpoints, & virtualenv
└── README.md                      # Dokumentasi proyek
```

---

## 🛠️ Persyaratan Library (Dependencies)

Library yang diperlukan untuk menjalankan seluruh alur proyek:

| Kategori | Library | Keterangan |
| :--- | :--- | :--- |
| **Data Collection** | `google-play-scraper` | Mengambil ulasan dari Google Play Store |
| **Data Manipulation** | `pandas`, `numpy` | Manipulasi tabel data, pembacaan CSV/JSON, operasi matriks |
| **NLP Preprocessing** | `nltk`, `Sastrawi` | Tokenisasi (punkt), kamus stopword & stemmer bahasa Indonesia |
| **Machine Learning** | `scikit-learn` | TF-IDF Vectorizer, Train/Test Split, 5-Fold CV, Logistic Regression |
| **Inter-Rater Reliability** | `statsmodels` | Pengujian Fleiss' Kappa untuk analisis kesepakatan anotator |
| **Boosting Model** | `catboost` | Model klasifikasi CatBoost (support CPU & GPU) |
| **Visualisasi** | `matplotlib` | Visualisasi Confusion Matrix |
| **Integrasi Sheet** | `gspread`, `google-auth` | Ekspor hasil kembali ke Google Sheets (opsional) |
| **Jupyter** | `ipykernel` | Menjalankan notebook di environment lokal |

### Cara Instalasi:

```bash
# Buat virtual environment (disarankan)
python -m venv venv

# Aktivasi virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install semua dependensi
pip install -r requirements.txt
```

---

## 🔄 Tahapan Pipeline Preprocessing & Analisis

1. **Pengumpulan Data (Data Collection)**
   - Mengambil ulasan MyTelkomsel dari Google Play Store (`com.telkomsel.telkomselcm`) atau memuat data berlabel dari Google Sheets.
2. **Tokenisasi (Tokenization)**
   - Penggabungan angka bintang (misal: `"1 bintang"` $\rightarrow$ `"bintang_1"`).
   - Pembersihan tanda baca dengan spasi agar kata tidak menempel.
   - Tokenisasi menggunakan `nltk.word_tokenize`.
3. **Normalisasi (Normalization)**
   - Normalisasi Unicode (`NFKC`) dan teks huruf kecil (*lowercase*).
   - Pemetaan emoji ke kata sentimen (contoh: 👍 $\rightarrow$ `"bagus"`, 😡 $\rightarrow$ `"marah"`).
   - Penanganan kata slang/alay via kamus lexicon daring (4.300+ entri) dan kamus khusus istilah telekomunikasi (contoh: *lemot* $\rightarrow$ *lambat*, *tsel* $\rightarrow$ *telkomsel*).
   - Penanganan reduplikasi (contoh: `"masuk2"` $\rightarrow$ `"masuk masuk"`).
4. **Pembersihan Kata Henti (Stopword Removal)**
   - Menggunakan `Sastrawi.StopWordRemover`.
   - Menjaga kata-kata negasi dan pembentuk frasa sentimen (*"tidak", "belum", "bukan", "jangan", "tapi", "bisa", "masih"*).
5. **Stemming**
   - Menggunakan `Sastrawi.Stemmer` yang dilengkapi dengan `stem_cache` dan `stem_manual`.
   - Menjaga token bernomor/ber-underscore (seperti `"bintang_1"`) agar tidak rusak.
6. **Analisis Bias Pelabelan (Inter-Rater Agreement)**
   - Menghitung **Fleiss' Kappa** antar 5 anotator manusia (*Bagus, Daniel, Liza, Cila, Missy*).
7. **Pemodelan & Evaluasi Sentimen**
   - Ekstraksi fitur menggunakan **TF-IDF (unigram & bigram)**.
   - Penyeimbangan bobot kelas (*class weights: balanced*) untuk mengatasi data tidak seimbang (*imbalance*).
   - Pelatihan **CatBoost Classifier** dengan validasi *early stopping*.
   - Evaluasi komparatif menggunakan **Stratified 5-Fold Cross Validation** bersama baseline **Logistic Regression**.

