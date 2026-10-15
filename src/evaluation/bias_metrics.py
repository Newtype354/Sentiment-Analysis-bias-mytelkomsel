"""
Inter-annotator agreement analysis (Fleiss' Kappa) to evaluate labeling bias.
"""

import pandas as pd
from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa


LABEL_MAP = {"negative": -1, "neutral": 0, "positive": 1}
DEFAULT_ANNOTATORS = ["Bagus", "Daniel", "Liza", "Cila", "Missy"]


def compute_fleiss_kappa(
    df: pd.DataFrame,
    annotators: list[str] = None,
    label_map: dict = None
) -> float:
    """
    Menghitung Fleiss' Kappa untuk mengukur kesepakatan antar-anotator manusia.
    Fleiss' Kappa > 0.80 umumnya menandakan hampir sepakat sempurna (almost perfect agreement).
    """
    if annotators is None:
        annotators = DEFAULT_ANNOTATORS
    if label_map is None:
        label_map = LABEL_MAP

    existing_annotators = [c for c in annotators if c in df.columns]
    if len(existing_annotators) < 2:
        raise ValueError(f"Dibutuhkan minimal 2 kolom anotator, ditemukan: {existing_annotators}")

    lab = df[existing_annotators].apply(
        lambda s: s.astype(str).str.strip().str.lower().map(label_map)
    )
    lab = lab.dropna().astype(int)

    table, _ = aggregate_raters(lab.values)
    kappa = fleiss_kappa(table)

    print(f"--- Evaluasi Bias Pelabelan ---")
    print(f"Jumlah anotator: {len(existing_annotators)} ({', '.join(existing_annotators)})")
    print(f"Jumlah sampel ulasan: {len(lab)}")
    print(f"Fleiss' Kappa: {kappa:.4f}")

    return kappa

