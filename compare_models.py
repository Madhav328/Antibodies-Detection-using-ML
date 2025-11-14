import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr

# Load ESM embeddings
X = np.load("X_features.npy")
y = np.load("y_reg.npy")

print("Loaded:", X.shape, y.shape)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Models to compare
models = {
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "Extra Trees": ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "SVR (RBF)": SVR(kernel="rbf"),
    "Linear Regression": LinearRegression(),
    "KNN (k=5)": KNeighborsRegressor(n_neighbors=5)
}

mse_vals = []
r2_vals = []
spearman_vals = []
names = list(models.keys())

# Train and evaluate
for name, model in models.items():
    print("Training:", name)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    spear = spearmanr(y_test, preds)[0]

    mse_vals.append(mse)
    r2_vals.append(r2)
    spearman_vals.append(spear)

# Normalize metrics
def normalize(arr):
    arr = np.array(arr)
    min_v, max_v = arr.min(), arr.max()
    return (arr - min_v) / (max_v - min_v + 1e-8)

norm_mse = 1 - normalize(mse_vals)      # lower mse = higher score
norm_r2 = normalize(r2_vals)
norm_spear = normalize(spearman_vals)

# Composite Score
composite_score = norm_mse + norm_r2 + norm_spear

# Plot
plt.figure(figsize=(12,6))
sns.set_style("whitegrid")

sns.barplot(x=names, y=composite_score, palette="viridis")

plt.title("Overall Model Performance (Composite Score)\nHigher = Better", fontsize=16)
plt.ylabel("Composite Score", fontsize=14)
plt.xticks(rotation=25, ha='right')
plt.tight_layout()

plt.savefig("model_comparison_single_plot.png", dpi=350)
plt.close()

print("\nSaved: model_comparison_single_plot.png")