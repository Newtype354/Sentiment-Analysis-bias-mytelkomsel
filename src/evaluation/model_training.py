"""
Model Training & Evaluation Pipeline: TF-IDF, CatBoost Classifier, and Logistic Regression baseline.
"""

import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier


LABEL_MAP = {"negative": -1, "neutral": 0, "positive": 1}
CLASSES = np.array([-1, 0, 1])


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Membersihkan ulasan kosong dan memetakan label target."""
    data = df.copy()
    data["ulasan_clean"] = data["ulasan_clean"].fillna("").astype(str)
    data = data[data["ulasan_clean"].str.strip() != ""].reset_index(drop=True)

    data["target"] = data["Summary"].astype(str).str.strip().str.lower().map(LABEL_MAP)
    data = data.dropna(subset=["target"]).reset_index(drop=True)
    data["target"] = data["target"].astype(int)
    return data


def train_and_evaluate(
    df: pd.DataFrame,
    show_plot: bool = True
) -> dict:
    """Melatih CatBoostClassifier dan LogisticRegression serta melakukan 5-fold CV."""
    data = prepare_data(df)
    X = data["ulasan_clean"]
    y = data["target"]

    # Split train-val-test
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.15, random_state=42, stratify=y_train_full
    )

    # Vectorization TF-IDF
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    X_tr_vec = vec.fit_transform(X_train).toarray()
    X_val_vec = vec.transform(X_val).toarray()
    X_te_vec = vec.transform(X_test).toarray()

    # Bobot kelas untuk menangani ketidakseimbangan (imbalance)
    weights = compute_class_weight("balanced", classes=CLASSES, y=y_train)
    class_weights = dict(zip(CLASSES, weights))

    task_type = "GPU" if shutil.which("nvidia-smi") else "CPU"
    print(f"CatBoost task_type: {task_type}")

    model_cb = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="MultiClass",
        iterations=500,
        learning_rate=0.05,
        depth=8,
        class_weights=class_weights,
        random_seed=42,
        verbose=100,
        task_type=task_type,
    )

    model_cb.fit(
        X_tr_vec,
        y_train,
        eval_set=(X_val_vec, y_val),
        early_stopping_rounds=50
    )

    y_pred = model_cb.predict(X_te_vec).ravel()
    report = classification_report(
        y_test, y_pred, labels=CLASSES,
        target_names=["Negative", "Neutral", "Positive"], digits=4
    )
    print("\n--- Classification Report (CatBoost on Test Set) ---")
    print(report)

    if show_plot:
        cm = confusion_matrix(y_test, y_pred, labels=CLASSES)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Neutral", "Positive"])
        disp.plot()
        plt.title("Confusion Matrix - CatBoost")
        plt.xlabel("Label Prediksi")
        plt.ylabel("Label Sebenarnya")
        plt.show()

    # Stratified 5-Fold Cross Validation
    print("\n--- Menjalankan Stratified 5-Fold Cross Validation ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_iter = model_cb.get_best_iteration() + 1
    skor_cb, skor_lr = [], []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X.values, y.values), 1):
        fold_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
        X_tr = fold_vec.fit_transform(X.values[tr_idx])
        X_te = fold_vec.transform(X.values[te_idx])

        w = compute_class_weight("balanced", classes=CLASSES, y=y.values[tr_idx])
        cb = CatBoostClassifier(
            loss_function="MultiClass", iterations=best_iter, learning_rate=0.05,
            depth=8, class_weights=dict(zip(CLASSES, w)), random_seed=42,
            verbose=0, task_type=task_type
        )
        cb.fit(X_tr.toarray(), y.values[tr_idx])
        skor_cb.append(f1_score(y.values[te_idx], cb.predict(X_te.toarray()).ravel(), average="macro"))

        lr = LogisticRegression(max_iter=2000, class_weight="balanced")
        lr.fit(X_tr, y.values[tr_idx])
        skor_lr.append(f1_score(y.values[te_idx], lr.predict(X_te), average="macro"))

    print(f"CatBoost  5-fold macro F1: {np.mean(skor_cb):.4f} ± {np.std(skor_cb):.4f}")
    print(f"LogReg    5-fold macro F1: {np.mean(skor_lr):.4f} ± {np.std(skor_lr):.4f}")

    return {
        "catboost_test_f1": f1_score(y_test, y_pred, average="macro"),
        "catboost_cv_mean": np.mean(skor_cb),
        "logreg_cv_mean": np.mean(skor_lr)
    }

