import pandas as pd

df = pd.read_csv("Real dataset.csv", sep=None, engine="python")

print("Raw columns:", df.columns.tolist())

# Fix BOM prefix in first column
df = df.rename(columns={df.columns[0]: "HC_Masked"})

print("Cleaned columns:", df.columns.tolist())
print("Rows loaded:", len(df))

rows = []

for i, row in df.iterrows():

    hc = row.get("HC_Masked", "")
    cdr3 = row.get("HCDR3_winning", "")
    lc = row.get("LC", "")

    # ----- HEAVY -----
    if isinstance(hc, str):
        if "[MASK]" in hc and isinstance(cdr3, str):
            heavy_seq = hc.replace("[MASK]", cdr3)
        else:
            heavy_seq = hc

        if isinstance(heavy_seq, str) and len(heavy_seq) > 20:
            rows.append({
                "id": f"heavy_{i}",
                "chain_type": "heavy",
                "sequence": heavy_seq
            })

    # ----- LIGHT -----
    if isinstance(lc, str) and len(lc) > 20:
        rows.append({
            "id": f"light_{i}",
            "chain_type": "light",
            "sequence": lc
        })

out = pd.DataFrame(rows)
out.to_csv("antibodies.csv", index=False)

print("Created antibodies.csv with", len(out), "sequences!")