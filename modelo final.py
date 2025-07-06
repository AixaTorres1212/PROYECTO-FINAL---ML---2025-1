import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import train_test_split

# ------------------ 1. Datos y zona gris ------------------
SEED = 42
LOW, HIGH = 0.35, 0.55
CONF_LIM = 0.10

X = pd.read_csv(
    r"C:/Users/axime/Desktop/proyecto_final/pruebas_finales/vae_embeddings.csv",
    index_col=0
).values
y = pd.read_csv(
    r"C:/Users/axime/Desktop/proyecto_final/pruebas_finales/y_labels.csv",
    index_col=0
).values.ravel().astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED
)

probs_rf_file = r"C:/Users/axime/Desktop/proyecto_final/probs_rf_test.csv"
probs_rf = pd.read_csv(probs_rf_file)["prob_RF"].values

mask_gris = (probs_rf >= LOW) & (probs_rf <= HIGH)
Xg, yg = X_test[mask_gris], y_test[mask_gris]
print(f"🔍 Ejemplos en zona gris TEST: {Xg.shape[0]} / {X_test.shape[0]}")

# ------------------ 2. Submodelo XGBoost ------------------
scale_pos = sum(yg == 0) / sum(yg == 1)
xgb = XGBClassifier(
    objective="binary:logistic",
    max_depth=3,
    learning_rate=0.1,
    n_estimators=50,
    gamma=0,
    subsample=0.8,
    scale_pos_weight=scale_pos,
    eval_metric="logloss",
    random_state=SEED,
)
xgb.fit(Xg, yg)

# ------------------ 3. Random Forest base ------------------
rf_final = joblib.load(
    r"C:/Users/axime/Desktop/proyecto_final/rf_vae_smote_opt.pkl"
)
prob_rf_test = rf_final.predict_proba(X_test)[:, 1]
pred_hibrido = (prob_rf_test >= 0.5).astype(int)

# ------------------ 4. Inferencia zona gris + prob_hibrido ------------------
prob_xgb_gris = xgb.predict_proba(Xg)[:, 1]
override = (prob_xgb_gris >= 0.5 + CONF_LIM) | (prob_xgb_gris <= 0.5 - CONF_LIM)
preds_xgb = (prob_xgb_gris >= 0.5).astype(int)

pred_hibrido[mask_gris] = np.where(override, preds_xgb, pred_hibrido[mask_gris])

# ▶️ Probabilidad híbrida final (para AUROC/AUPRC híbridos)
prob_hibrido = prob_rf_test.copy()
prob_hibrido[mask_gris] = np.where(override, prob_xgb_gris, prob_rf_test[mask_gris])

# ------------------ 5. Reporte ------------------
print("\n=== Reporte híbrido final (XGB zona gris) ===")
print(classification_report(y_test, pred_hibrido, digits=4))

print("AUROC híbrido :", round(roc_auc_score(y_test, prob_hibrido), 4))
print("AUPRC híbrido :", round(average_precision_score(y_test, prob_hibrido), 4))

print("Confusion:\n", confusion_matrix(y_test, pred_hibrido))

# ------------------ 6. Guardado ------------------
out_dir = r"C:/Users/axime/Desktop/proyecto_final/hibrido_xgb_zonagris"
import os
os.makedirs(out_dir, exist_ok=True)
joblib.dump(xgb, os.path.join(out_dir, "xgb_submodelo.pkl"))
with open(os.path.join(out_dir, "zonagris_info.txt"), "w") as f:
    f.write(f"LOW={LOW}\nHIGH={HIGH}\nCONF_LIM={CONF_LIM}\n")
