# rf_vae_smote_hyperopt_3folds.py
import warnings, joblib, numpy as np, pandas as pd
from pathlib import Path
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (classification_report, roc_auc_score,
                             average_precision_score, confusion_matrix)

warnings.filterwarnings("ignore")

# ---------- 1. Cargar datos ----------
X = pd.read_csv(
        r"C:\Users\axime\Desktop\proyecto_final\pruebas_finales\vae_embeddings.csv",
        index_col=0).values
y = pd.read_csv(
        r"C:\Users\axime\Desktop\proyecto_final\pruebas_finales\y_labels.csv",
        index_col=0).values.ravel().astype(int)

# ---------- 2. Split 80/20 ----------
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42)

# ---------- 3. Construir Pipeline (SMOTE + RF) ----------
pipe = Pipeline(steps=[
        ("smote", SMOTE(random_state=42)),
        ("rf", RandomForestClassifier(
                n_estimators=100,      # fijo para ir rápido
                random_state=42,
                n_jobs=-1))
])

# ---------- 4. Espacio de hiperparámetros (se refieren al paso "rf__") ----------
param_dist = {
    "rf__max_depth":         [10, 20, None],
    "rf__max_features":      ["sqrt", 0.5],
    "rf__min_samples_leaf":  [1, 2],
    "rf__min_samples_split": [2, 4]
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=10,                 # ↔ equilibrio velocidad/exploración
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=42)

# ---------- 5. Ajustar ----------
search.fit(X_train, y_train)
best_model = search.best_estimator_

# ---------- 6. Evaluar en test ----------
probs = best_model.predict_proba(X_test)[:, 1]
preds = (probs >= 0.5).astype(int)

print("\n🏆  Mejores hiperparámetros:")
for k, v in search.best_params_.items():
    print(f"   {k}: {v}")
print(f"AUCPR CV (mean): {search.best_score_:.4f}")

print("\n=== Reporte en Test ===")
print(classification_report(y_test, preds, digits=4))
print("AUROC :", round(roc_auc_score(y_test, probs), 4))
print("AUCPR :", round(average_precision_score(y_test, probs), 4))
print("Confusion:\n", confusion_matrix(y_test, preds))

# ---------- 7. Guardar modelo ----------
out = Path(r"C:\Users\axime\Desktop\proyecto_final\rf_vae_smote_opt.pkl")
joblib.dump(best_model, out)
print(f"\n✅ Modelo guardado en {out}")
