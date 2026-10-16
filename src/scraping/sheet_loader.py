"""
Module for loading labeled dataset from Google Sheets.
"""
import pandas as pd
DEFAULT_SHEET_ID = "1JVpzO2-EAivK-HwRGoiLHcOTJWzX94UV1T1y1Xz-7Jg"
DEFAULT_GID = "1412720530"


def load_from_google_sheets(sheet_id: str = DEFAULT_SHEET_ID,gid: str = DEFAULT_GID,output_path: str = "data/raw/ulasan_mytelkom_2026.csv") -> pd.DataFrame:
    """
    Mengunduh data ulasan berlabel dari Google Sheets yang diekspor sebagai CSV.
    """
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
    print(f"Mengunduh data dari: {url}")
    df = pd.read_csv(url)
    df["ulasan"] = df["ulasan"].fillna("").astype(str)
    if output_path:
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"Data disimpan ke: {output_path} ({len(df)} baris)")

    return df


if __name__ == "__main__":
    load_from_google_sheets()

