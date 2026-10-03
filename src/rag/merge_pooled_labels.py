# save as: evaluation/merge_pooled_labels.py
import pandas as pd
from pathlib import Path

EVAL_DIR = Path("evaluation")

existing = pd.read_csv(EVAL_DIR / "gold_relevance.csv")
new      = pd.read_csv(EVAL_DIR / "pooled_labeling_template.csv")

# keep only rows with a label filled in
new_labeled = new[
    new["relevance"].notna() &
    (new["relevance"].astype(str).str.strip() != "")
][["query_id","incident_id","relevance"]].copy()

new_labeled["relevance"] = pd.to_numeric(
    new_labeled["relevance"], errors="coerce"
).fillna(0).astype(int)

# merge
combined = pd.concat([existing, new_labeled], ignore_index=True)
combined = combined.drop_duplicates(
    subset=["query_id","incident_id"],
    keep="last"
)
combined.to_csv(EVAL_DIR / "gold_relevance.csv", index=False)
print(f"Updated gold_relevance.csv: {len(combined):,} rows")