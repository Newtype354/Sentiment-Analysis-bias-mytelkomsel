"""
Module for scraping Google Play Store reviews for MyTelkomsel.
"""
from google_play_scraper import reviews, Sort, app as get_app_info
import pandas as pd

def scrape_mytelkomsel_reviews(app_id: str = "com.telkomsel.telkomselcm",count: int = 500,year: int = 2026,output_path: str = "data/raw/ulasan_mytelkom_2026.csv") -> pd.DataFrame:
    """
    Mengambil ulasan MyTelkomsel dari Google Play Store dan memfilter berdasarkan tahun.
    """
    print(f"Mengambil info aplikasi: {app_id}...")
    _ = get_app_info(app_id, lang="id", country="id")
    print(f"Mengambil {count} ulasan terbaru...")
    hasil, _ = reviews(app_id,lang="id",country="id",sort=Sort.NEWEST,count=count,)
    df_raw = pd.DataFrame(hasil)[["reviewId", "userName", "content", "score", "at", "reviewCreatedVersion"]]
    df_raw.columns = ["id_ulasan", "nama_pengguna", "ulasan", "rating", "tanggal", "versi_aplikasi_ulasan"]
    df_raw["tanggal"] = pd.to_datetime(df_raw["tanggal"])
    df = df_raw[df_raw["tanggal"].dt.year == year].copy()
    print(f"Berhasil mengambil {len(df)} ulasan untuk tahun {year}.")
    if output_path:
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"Data disimpan ke: {output_path}")

    return df


if __name__ == "__main__":
    scrape_mytelkomsel_reviews()

