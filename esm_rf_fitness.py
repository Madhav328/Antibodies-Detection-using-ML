import pandas as pd
import numpy as np
import torch
import esm
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, roc_auc_score
from scipy.stats import spearmanr

# 1. Load data (auto-detect delimiter) and fix BOM
df = pd.read_csv("Real dataset.csv", sep=None, engine="python")
df = df.rename(columns={df.columns[0]: "HC_Masked"})

df = df.dropna(subset=["HC_Masked", "HCDR3_winning", "HCDR3_losing", "fitness_winning", "fitness_losing", "LC"])

df["fitness_winning"] = pd.to_numeric(df["fitness_winning"], errors="coerce")
df["fitness_losing"] = pd.to_numeric(df["fitness_losing"], errors="coerce")
df = df.dropna(subset=["fitness_winning", "fitness_losing"])

print("Rows loaded:", len(df))

# 2. Build winning heavy chain
def build_heavy(seq_masked, cdr3):
    if not isinstance(seq_masked, str) or not isinstance(cdr3, str):
        return None
    return seq_masked.replace("[MASK]", cdr3)

df["HC_win"] = df.apply(lambda r: build_heavy(r["HC_Masked"], r["HCDR3_winning"]), axis=1)
df = df.dropna(subset=["HC_win", "LC", "HCDR3_winning"])
print("Rows after HC_win:", len(df))

# 3. Load ESM model
print("Loading ESM-2 t6_8M...")
model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()
device = torch.device("cpu")
model = model.to(device)

embedding_cache = {}

def get_esm_embedding(seq):
    if not isinstance(seq, str) or len(seq) < 5:
        return None
    if seq in embedding_cache:
        return embedding_cache[seq]
    seq_clean = "".join([aa for aa in seq if aa.isalpha()])
    data = [("seq", seq_clean)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)
    with torch.no_grad():
        out = model(tokens, repr_layers=[6], return_contacts=False)
        reps = out["representations"][6][0]
    vec = reps[1:-1].mean(0).cpu().numpy()
    embedding_cache[seq] = vec
    return vec

# 4. Build feature matrix
features = []
targets_reg = []
targets_cls = []

print("Computing embeddings...")
for _, row in tqdm(df.iterrows(), total=len(df)):
    hc = row["HC_win"]
    lc = row["LC"]
    cdr3 = row["HCDR3_winning"]

    emb_hc = get_esm_embedding(hc)
    emb_lc = get_esm_embedding(lc)
    emb_cdr3 = get_esm_embedding(cdr3)

    if emb_hc is None or emb_lc is None or emb_cdr3 is None:
        continue

    feat = np.concatenate([emb_hc, emb_lc, emb_cdr3])
    features.append(feat)
    targets_reg.append(row["fitness_winning"])
    label = 1 if row["fitness_winning"] > row["fitness_losing"] else 0
    targets_cls.append(label)

X = np.vstack(features)
y_reg = np.array(targets_reg, dtype=float)
y_cls = np.array(targets_cls, dtype=int)

print("Feature matrix:", X.shape)

X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = train_test_split(
    X, y_reg, y_cls, test_size=0.2, random_state=42
)

# 5. Regression
print("\n=== RF Regression (fitness_winning) ===")
rf_reg = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
rf_reg.fit(X_train, y_reg_train)
y_reg_pred = rf_reg.predict(X_test)

mse = mean_squared_error(y_reg_test, y_reg_pred)
r2 = r2_score(y_reg_test, y_reg_pred)
spearman, _ = spearmanr(y_reg_test, y_reg_pred)

print(f"MSE: {mse:.4f}")
print(f"R^2: {r2:.4f}")
print(f"Spearman: {spearman:.4f}")

# 6. Classification
print("\n=== RF Classification (winner vs loser) ===")
rf_cls = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
rf_cls.fit(X_train, y_cls_train)
y_cls_pred = rf_cls.predict(X_test)
y_cls_proba = rf_cls.predict_proba(X_test)[:,1]

acc = accuracy_score(y_cls_test, y_cls_pred)
roc = roc_auc_score(y_cls_test, y_cls_proba)

print(f"Accuracy: {acc:.4f}")
print(f"ROC AUC: {roc:.4f}")

# 7. Save features for plotting
np.save("X_features.npy", X)
np.save("y_reg.npy", y_reg)
np.save("y_cls.npy", y_cls)

print("Saved X_features.npy, y_reg.npy, y_cls.npy")