import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc
from scipy.stats import spearmanr

# Load feature matrix + labels
X = np.load("X_features.npy")
y_reg = np.load("y_reg.npy")
y_cls = np.load("y_cls.npy")

print("Loaded:", X.shape, y_reg.shape, y_cls.shape)

# =============== PCA  ==================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8,6))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y_cls, palette="viridis", s=10)
plt.title("PCA of ESM Embeddings (Winner vs Loser)")
plt.savefig("plot_pca.png", dpi=300)
plt.close()

# =============== t-SNE  ==================
from sklearn.manifold import TSNE

print("Running t-SNE... (this may take 1–3 minutes)")

tsne = TSNE(
    n_components=2,
    perplexity=30,
    init='pca',
    learning_rate='auto'
)

X_tsne = tsne.fit_transform(X)

plt.figure(figsize=(8,6))
sns.scatterplot(
    x=X_tsne[:,0], 
    y=X_tsne[:,1], 
    hue=y_cls, 
    palette="coolwarm", 
    s=10
)
plt.title("t-SNE of ESM Embeddings")
plt.savefig("plot_tsne.png", dpi=300)
plt.close()

# =============== Regression Scatter ==================
X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_train, y_train_reg)
y_pred_reg = reg.predict(X_test)

plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test_reg, y=y_pred_reg, s=10)
plt.plot([min(y_test_reg), max(y_test_reg)], [min(y_test_reg), max(y_test_reg)], "r--")
plt.xlabel("Actual Fitness")
plt.ylabel("Predicted Fitness")
plt.title("Regression: Actual vs Predicted")
plt.savefig("plot_regression.png", dpi=300)
plt.close()

# Spearman heatmap
corr, _ = spearmanr(y_test_reg, y_pred_reg)
plt.figure(figsize=(3,3))
sns.heatmap([[corr]], annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Spearman Correlation")
plt.savefig("plot_spearman_heatmap.png", dpi=300)
plt.close()

# =============== Classification Plots ==================
X_train, X_test, y_train_cls, y_test_cls = train_test_split(X, y_cls, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train_cls)
y_pred_cls = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:,1]

# Confusion Matrix
cm = confusion_matrix(y_test_cls, y_pred_cls)
plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix (Winner vs Loser)")
plt.savefig("plot_confusion_matrix.png", dpi=300)
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test_cls, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
plt.plot([0,1], [0,1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig("plot_roc_curve.png", dpi=300)
plt.close()

print("All plots saved successfully!")