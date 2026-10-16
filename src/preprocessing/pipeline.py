"""
Text Preprocessing Pipeline: Tokenization -> Normalization -> Stopwords -> Stemming.
"""

import json
import os
import re
import unicodedata
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


# Pastikan resource NLTK terunduh
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")


ANGKA = {"satu": "1", "dua": "2", "tiga": "3", "empat": "4", "lima": "5"}


def gabung_bintang(teks: str) -> str:
    """Menggabungkan frasa bintang (misal: 'bintang 1', '1 bintang' -> 'bintang_1')."""
    pola_angka = r"(satu|dua|tiga|empat|lima|[1-5])"
    teks = re.sub(
        rf"\b{pola_angka}\s*bintang\b",
        lambda m: "bintang_" + ANGKA.get(m.group(1), m.group(1)),
        teks
    )
    teks = re.sub(
        rf"\bbintang\s*{pola_angka}\b",
        lambda m: "bintang_" + ANGKA.get(m.group(1), m.group(1)),
        teks
    )
    return teks


def tokenize(teks: str) -> list:
    """Tokenisasi teks menggunakan NLTK word_tokenize setelah pembersihan tanda baca."""
    teks = str(teks).lower()
    teks = gabung_bintang(teks)
    for tanda in [".", ",", ";", ":", "?", "!"]:
        teks = teks.replace(tanda, " ")
    return word_tokenize(teks)


def build_normalization_dict() -> tuple[dict, dict]:
    """Membangun kamus slang (alay), kamus manual istilah telkomsel, dan kamus emoji."""
    url_kamus = "https://raw.githubusercontent.com/nasalsabila/kamus-alay/master/colloquial-indonesian-lexicon.csv"
    try:
        df_kamus = pd.read_csv(url_kamus).dropna(subset=["slang", "formal"])
        kamus_slang = dict(zip(
            df_kamus["slang"].astype(str).str.lower().str.strip(),
            df_kamus["formal"].astype(str).str.lower().str.strip()
        ))
    except Exception:
        kamus_slang = {}
    kamus_manual = {
        "gk": "tidak", "ga": "tidak", "gak": "tidak", "ngak": "tidak", "nggak": "tidak", "tdk": "tidak",
        "yg": "yang", "udh": "sudah", "udah": "sudah", "sdh": "sudah", "blm": "belum", "belom": "belum",
        "bgt": "banget", "sy": "saya", "aq": "aku", "gw": "saya", "gue": "saya",
        "dgn": "dengan", "dg": "dengan", "utk": "untuk", "tp": "tapi", "krn": "karena", "karna": "karena",
        "trs": "terus", "jd": "jadi", "jg": "juga", "lg": "lagi", "bs": "bisa", "bsa": "bisa", "sm": "sama",
        "knp": "kenapa", "gmn": "bagaimana", "gimana": "bagaimana", "kalo": "kalau", "klo": "kalau",
        "dr": "dari", "org": "orang", "hrs": "harus", "msh": "masih", "aja": "saja", "aj": "saja",
        "pake": "pakai", "pakek": "pakai", "apk": "aplikasi", "app": "aplikasi", "apl": "aplikasi",
        "tsel": "telkomsel", "kouta": "kuota", "qouta": "kuota", "signal": "sinyal", "singnal": "sinyal",
        "min": "admin", "mimin": "admin", "gpp": "tidak apa apa", "gapapa": "tidak apa apa",
        "mantul": "mantap betul", "makasih": "terima kasih", "mksh": "terima kasih", "trims": "terima kasih",
        "thx": "terima kasih", "tq": "terima kasih", "ok": "oke", "okay": "oke", "bgs": "bagus",
        "lemot": "lambat", "lelet": "lambat", "erorr": "error", "pket": "paket", "eror": "error",
        "ngetod": "ngentod", "taaiiiiiiii": "tahi", "pol": "banget", "pelitt": "pelit", "anjig": "anjing",
        "pukii": "puki", "provder": "provider", "njingg": "anjing", "berberapa": "beberapa",
        "loadingnya": "loading", "bitang": "bintang", "sangtlah": "sangat", "semuaa": "semua",
        "jaringannya": "jaringan", "indihomenya": "indihome", "ajg": "anjing", "relog": "login",
        "sebeluh": "sebelah", "kont0l": "kontol", "byar": "bayar", "fix": "perbaiki", "mausuk": "masuk",
        "mahalin": "mahal", "stbill": "stabil", "anying": "anjing", "gbsa": "tidak", "desedot": "disedot",
        "gblok": "goblok", "bug": "masalah", "mura": "murah", "mkasih": "terima kasih", "lag": "lambat",
        "samsek": "sama sekali", "enggak": "tidak", "engga": "tidak", "ngga": "tidak", "kagak": "tidak",
        "kaga": "tidak", "engak": "tidak", "gajelas": "tidak jelas", "gabisa": "tidak bisa",
        "gada": "tidak ada", "gaada": "tidak ada", "gamau": "tidak mau", "gapernah": "tidak pernah",
        "mb": "mb", "gb": "gb", "kelemotan": "lambat", "lemotnya": "lambat", "lemoth": "lambat",
        "kuoto": "kuota", "membatu": "membantu", "memuaskn": "memuaskan", "bsgus": "bagus", "sangst": "sangat"
    }
    kamus_normalisasi = {**kamus_slang, **kamus_manual}

    kamus_emoji = {
        "👍": " bagus ", "👎": " jelek ", "🥰": " suka ", "😍": " suka ", "❤": " suka ",
        "🙏": " terima kasih ", "😡": " marah ", "🤬": " marah ", "😭": " sedih ",
        "😞": " kecewa ", "🤢": " jijik ", "🙄": " kesal ", "😤": " kesal ", "😠": " marah ",
        "😢": " sedih ", "💩": " jelek ", "😊": " senang ", "😁": " senang ", "🔥": " mantap ", "💯": " mantap "
    }

    return kamus_normalisasi, kamus_emoji


def normalize_tokens(tokens: list, kamus_normalisasi: dict, kamus_emoji: dict) -> list:
    """Normalisasi token: unicode NFKC, emoji, url, reduksi karakter berulang, dan pemetaan kamus."""
    normalized_list = []
    for token in tokens:
        token = unicodedata.normalize("NFKC", token).lower()
        for emoji_char, rep in kamus_emoji.items():
            token = token.replace(emoji_char, rep)

        token = re.sub(r"http\S+|www\S+", "", token)
        token = token.replace("-", " ")
        token = re.sub(r"[^a-z0-9_\s]", " ", token)
        token = re.sub(r"\s+", " ", token).strip()
        token = re.sub(r"([a-z])\1{2,}", r"\1", token)

        sub_tokens = token.split()
        for sub_token in sub_tokens:
            match_rep = re.match(r"([a-z]{2,})2$", sub_token)
            if match_rep:
                base_word = match_rep.group(1)
                normalized_list.extend([base_word, base_word])
                continue

            first_norm = kamus_normalisasi.get(sub_token, sub_token)
            for part in first_norm.split():
                normalized_list.append(kamus_normalisasi.get(part, part))

    return [t for t in normalized_list if t]


def get_stopwords() -> set:
    """Mengambil stopwords Sastrawi dengan mempertahankan kata negasi dan sentimen."""
    factory = StopWordRemoverFactory()
    sw = set(factory.get_stop_words())
    kata_dipertahankan = {
        "tidak", "belum", "bukan", "jangan", "bisa", "tapi", "tetapi",
        "namun", "tanpa", "tolong", "dapat", "harus", "masih", "hanya", "agak"
    }
    sw -= kata_dipertahankan
    return sw


def build_stemmer():
    """Inisialisasi stemmer Sastrawi dan pengecualian kata manual."""
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    stem_cache = {}
    stem_manual = {
        "berikan": "beri", "kurangin": "kurang", "pengecekan": "cek",
        "sebelumnya": "sebelum", "perusahan": "perusahaan", "perusahaan": "perusahaan",
        "bekasi": "bekasi", "bali": "bali", "maluku": "maluku",
    }

    def stem_word(word: str) -> str:
        if word in stem_manual:
            return stem_manual[word]
        if "_" in word or any(ch.isdigit() for ch in word):
            return word
        if word not in stem_cache:
            stem_cache[word] = stemmer.stem(word)
        return stem_cache[word]

    return stem_word


def run_full_preprocessing(
    df: pd.DataFrame,
    output_clean_path: str = "data/processed/ulasan_Clean.json"
) -> list[dict]:
    """Menjalankan keseluruhan alur preprocessing pada DataFrame."""
    kamus_normalisasi, kamus_emoji = build_normalization_dict()
    stopwords = get_stopwords()
    stem_fn = build_stemmer()

    cleaned_data = []
    print("Memproses tokenisasi, normalisasi, stopword, dan stemming...")

    for _, row in df.iterrows():
        raw_text = str(row.get("ulasan", ""))
        tokens_asli = tokenize(raw_text)
        tokens_norm = normalize_tokens(tokens_asli, kamus_normalisasi, kamus_emoji)
        tokens_filtered = [k for k in tokens_norm if k.lower() not in stopwords]
        tokens_stemmed = [s for s in (stem_fn(t) for t in tokens_filtered) if s.strip()]

        item = row.to_dict()
        item["tokens_asli"] = tokens_asli
        item["tokens_stemmed"] = tokens_stemmed
        item["ulasan_clean"] = " ".join(tokens_stemmed)
        cleaned_data.append(item)

    if output_clean_path:
        os.makedirs(os.path.dirname(output_clean_path), exist_ok=True)
        with open(output_clean_path, "w", encoding="utf-8") as f:
            json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
        print(f"Data bersih disimpan di: {output_clean_path}")

    return cleaned_data

