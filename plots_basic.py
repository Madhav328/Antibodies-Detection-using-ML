import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("antibodies_developability.csv")

# CDR3 length
df["cdr3_len"] = df["cdr3"].fillna("").apply(len)

# 1. Histogram of CDR3 lengths
plt.figure(figsize=(8,5))
plt.hist(df["cdr3_len"], bins=20)
plt.xlabel("CDR3 length (aa)")
plt.ylabel("Count")
plt.title("Distribution of CDR3 lengths")
plt.tight_layout()
plt.savefig("plot_cdr3_length_hist.png", dpi=300)
plt.close()

# 2. Boxplot of instability index
plt.figure(figsize=(5,5))
df["instability_index"].dropna().plot(kind="box")
plt.ylabel("Instability index")
plt.title("Instability of antibody sequences")
plt.tight_layout()
plt.savefig("plot_instability_box.png", dpi=300)
plt.close()

# 3. Scatter: hydropathy vs instability
plt.figure(figsize=(7,5))
plt.scatter(df["hydropathy_gravy"], df["instability_index"], s=5, alpha=0.4)
plt.xlabel("GRAVY hydropathy (more negative = more soluble)")
plt.ylabel("Instability index (<40 usually stable)")
plt.title("Hydropathy vs Instability")
plt.tight_layout()
plt.savefig("plot_hydro_vs_instability.png", dpi=300)
plt.close()

print("Saved basic plots:")
print(" - plot_cdr3_length_hist.png")
print(" - plot_instability_box.png")
print(" - plot_hydro_vs_instability.png")