import numpy as np 
import pandas as pd
from collections import Counter, defaultdict
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix
import random
from imblearn.over_sampling import SMOTE     
import os

# --------------------------- Reproducibilidad ---------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# --------------------------- Árbol CART mínimo ---------------------------
class SimpleTree:
    class Node:
        __slots__ = ("feat","val","left","right","pred")
        def __init__(self,pred):
            self.feat=None; self.val=None
            self.left=None; self.right=None
            self.pred=pred
    def __init__(self,max_depth, min_samples_split, max_features):
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.max_features=max_features
        self.root=None
    @staticmethod
    def _gini(y):
        m=len(y)
        if m==0: return 0
        _,cnt=np.unique(y,return_counts=True)
        p=cnt/m
        return 1.0-np.sum(p**2)
    def _best_split(self,X,y,idx):
        Sg=self._gini(y[idx]); best_gain=0; best=None
        feats=np.random.choice(X.shape[1], self.max_features, replace=False)
        for f in feats:
            sidx=idx[np.argsort(X[idx,f])]
            y_sorted=y[sidx]
            left_cnt=defaultdict(int); right_cnt=Counter(y_sorted)
            n_left=0; n_tot=len(sidx)
            for i in range(1,n_tot):
                cls=y_sorted[i-1]
                left_cnt[cls]+=1; right_cnt[cls]-=1; n_left+=1
                if X[sidx[i],f]==X[sidx[i-1],f]: continue
                n_right=n_tot-n_left
                gl=1-np.sum([(c/n_left)**2 for c in left_cnt.values()])
                gr=1-np.sum([(c/n_right)**2 for c in right_cnt.values()])
                gain=Sg-(n_left/n_tot)*gl-(n_right/n_tot)*gr
                if gain>best_gain:
                    best_gain=gain
                    best=(f,(X[sidx[i-1],f]+X[sidx[i],f])/2)
        return best_gain,best
    def _build(self,X,y,idx,depth):
        pred=Counter(y[idx]).most_common(1)[0][0]
        node=self.Node(pred)
        if depth>=self.max_depth or len(idx)<self.min_samples_split:
            return node
        gain,best=self._best_split(X,y,idx)
        if best is None or gain==0: return node
        f,val=best
        mask=X[idx,f]<val
        left_idx=idx[mask]; right_idx=idx[~mask]
        if len(left_idx)==0 or len(right_idx)==0: return node
        node.feat,node.val=f,val
        node.left=self._build(X,y,left_idx,depth+1)
        node.right=self._build(X,y,right_idx,depth+1)
        return node
    def fit(self,X,y,idx):
        self.root=self._build(X,y,idx,0)
    def _pred_row(self,x,node):
        if node.left is None: return node.pred
        return self._pred_row(x,node.left) if x[node.feat]<node.val else self._pred_row(x,node.right)
    def predict(self,X):
        return np.array([self._pred_row(x,self.root) for x in X])

# --------------------------- Random Forest manual ---------------------------
class RandomForestScratch:
    def __init__(self, n_estimators=100, max_depth=5,
                 min_samples_split=2, min_samples_leaf=1,
                 max_features="sqrt", bootstrap=True,
                 class_weight=None, random_state=None):
        self.n_estimators=n_estimators
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.min_samples_leaf=min_samples_leaf
        self.max_features=max_features
        self.bootstrap=bootstrap
        self.class_weight=class_weight
        self.random_state=random_state
        self.trees=[]
        if random_state is not None:
            np.random.seed(random_state)
    def _sample_weights(self,y):
        if self.class_weight!='balanced': return np.ones_like(y,float)
        cnt=np.bincount(y); total=len(y)
        w=total/(len(cnt)*cnt); return w[y]
    def fit(self, X, y):
        n, d = X.shape
        mtry = int(np.log2(d)) if self.max_features == "log2" else int(np.sqrt(d))
        self.trees = []
        sample_weights = self._sample_weights(y)

        for i in range(self.n_estimators):
            if self.bootstrap:
                idx = np.random.choice(np.arange(n), size=n, replace=True)
            else:
                idx = np.random.permutation(n)

            tree = SimpleTree(self.max_depth, self.min_samples_split, mtry)
            tree.fit(X, y, idx)
            self.trees.append(tree)
            print(f"      Árbol {i+1}/{self.n_estimators} entrenado", end="\r")

        print(f"✅ Entrenamiento de {self.n_estimators} árboles completado")
        return self
    def predict(self,X):
        votes=np.vstack([t.predict(X) for t in self.trees])
        return np.apply_along_axis(lambda col: Counter(col).most_common(1)[0][0],0,votes)
    def predict_proba(self,X):
        votes=np.vstack([t.predict(X) for t in self.trees])
        return votes.mean(axis=0)

def print_tree(node, depth=0):
    prefix = "│   " * depth + ("├── " if depth > 0 else "")
    if node.left is None:
        print(f"{prefix}Predicción: {node.pred}")
    else:
        print(f"{prefix}if X[{node.feat}] < {node.val:.4f}:")
        print_tree(node.left, depth + 1)
        print(f"{'│   ' * depth}else:")
        print_tree(node.right, depth + 1)


# --------------------------- Pipeline completo ---------------------------


# --------------------------- Cargar datos ---------------------------
X = pd.read_csv(r"C:\Users\axime\Desktop\proyecto_final\pruebas_finales\vae_embeddings.csv",
                index_col=0).values
y = pd.read_csv(r"C:\Users\axime\Desktop\proyecto_final\pruebas_finales\y_labels.csv",
                index_col=0).values.ravel().astype(int)

# Split 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED)

# --------------------------- Hiperparámetros óptimos ---------------------------
params = dict(
    n_estimators=100,
    max_depth=20,
    min_samples_split=4,
    min_samples_leaf=1,
    max_features="sqrt",
    bootstrap=True,
    class_weight=None,
    random_state=SEED
)

# ---------------------- VALIDACIÓN CRUZADA CON SMOTE (3 folds) ----------------------
auprc_scores, auroc_scores = [], []
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

print("\n🔄 Validación cruzada 3-fold con SMOTE")
for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), 1):
    # ① Smote solo en cada fold de entrenamiento
    sm = SMOTE(random_state=SEED)
    X_tr_res, y_tr_res = sm.fit_resample(X_train[tr_idx], y_train[tr_idx])

    rf = RandomForestScratch(**params)
    rf.fit(X_tr_res, y_tr_res)

    prob_val = rf.predict_proba(X_train[va_idx])
    auprc_scores.append(average_precision_score(y_train[va_idx], prob_val))
    auroc_scores.append(roc_auc_score(y_train[va_idx], prob_val))

    print(f"   Fold {fold}/3  AUCPR={auprc_scores[-1]:.4f}  AUROC={auroc_scores[-1]:.4f}")

print("\n✅ Validación terminada")
print(f"CV AUCPR (3-fold): {np.mean(auprc_scores):.4f} ± {np.std(auprc_scores):.4f}")
print(f"CV AUROC (3-fold): {np.mean(auroc_scores):.4f} ± {np.std(auroc_scores):.4f}")

# ----------------------------- ENTRENAMIENTO FINAL -----------------------------
sm = SMOTE(random_state=SEED)
X_tr_res, y_tr_res = sm.fit_resample(X_train, y_train)

rf_final = RandomForestScratch(**params).fit(X_tr_res, y_tr_res)

# ----------------------------- EVALUACIÓN TEST -----------------------------
prob_test = rf_final.predict_proba(X_test)
thr = 0.42      # mismo umbral
pred_test = (prob_test >= thr).astype(int)

df_probs = pd.DataFrame({
    "prob_RF": prob_test,
    "y_true":  y_test,
    "pred":    pred_test,
    "acierto": pred_test == y_test
})
df_probs.to_csv("probs_rf_test.csv", index=False)
print("\n✅ Probabilidades guardadas en probs_rf_test.csv")

print("\n=== Reporte Final (umbral 0.42) ===")
print(classification_report(y_test, pred_test, digits=4))
print("AUROC :", round(roc_auc_score(y_test, prob_test), 4))
print("AUCPR :", round(average_precision_score(y_test, prob_test), 4))
print("Matriz de Confusión:\n", confusion_matrix(y_test, pred_test))