import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import re

df = pd.read_csv("antibodies_with_cdr.csv")

VALID_AA = "ACDEFGHIKLMNPQRSTVWY"

def clean_sequence(seq):
    if not isinstance(seq, str):
        return ""
    seq = seq.upper()
    # remove anything not A–Z
    seq = re.sub(r"[^A-Z]", "", seq)
    # keep only valid amino acids
    return "".join([aa for aa in seq if aa in VALID_AA])

pI_list = []
instability_list = []
gravy_list = []

for _, row in df.iterrows():
    raw_seq = row["sequence"]
    seq = clean_sequence(raw_seq)

    if len(seq) < 5:
        pI_list.append(None)
        instability_list.append(None)
        gravy_list.append(None)
        continue

    pa = ProteinAnalysis(seq)
    pI_list.append(pa.isoelectric_point())
    instability_list.append(pa.instability_index())
    gravy_list.append(pa.gravy())

df["pI"] = pI_list
df["instability_index"] = instability_list
df["hydropathy_gravy"] = gravy_list

df.to_csv("antibodies_developability.csv", index=False)
print("Saved antibodies_developability.csv with", len(df), "rows!")