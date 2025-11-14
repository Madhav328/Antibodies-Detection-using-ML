import pandas as pd
import re

# Read the sequences we just built
df = pd.read_csv("antibodies.csv")

def extract_cdr3(seq):
    """
    Very simple CDR3 finder:
    looks for C.....W pattern (3–30 aa between).
    Works for both heavy and light chains in most cases.
    """
    if not isinstance(seq, str):
        return ""
    match = re.search(r"C([A-Z]{3,30})W", seq)
    return match.group(1) if match else ""

cdr3_list = []

for seq in df["sequence"]:
    cdr3_list.append(extract_cdr3(seq))

df["cdr3"] = cdr3_list

df.to_csv("antibodies_with_cdr.csv", index=False)
print("Saved antibodies_with_cdr.csv with", len(df), "rows!")