"""Analyse how SMB page_texture classes affect OMR model performance (OMR-NED)."""

import os
import pandas as pd
from dotenv import load_dotenv
from datasets import load_dataset

load_dotenv()
token = os.environ.get("HF_TOKEN")

# -- 1. Load SMB dataset and extract page_texture per sample ------------------
print("Loading SMB dataset �")
ds = load_dataset("PRAIG/SMB", split="test", token=token)
print(f"  {len(ds)} samples, columns: {ds.column_names}")

# Build sample_id ? page_texture mapping (dataset is ordered; index i ? sample_XXXXXX)
texture_map = {}
for i, item in enumerate(ds):
    texture = item.get("page_texture")
    if texture is not None:
        texture_map[f"sample_{i:06d}"] = texture

print(f"  page_texture classes found: {sorted(set(texture_map.values()))}")
print(f"  texture_map sample: {list(texture_map.items())[:3]}")

# -- 2. Load evaluation CSVs --------------------------------------------------
CSV_FILES = {
    # "Audiveris": "generated/audiveris_trimmed.csv",
    "AudiverisScaled": "generated/audiveris_scaled_trimmed.csv",
    "HOMR":      "generated/homr_trimmed.csv",
    "oemer":     "generated/oemer_trimmed.csv",
}

def load_eval(path: str, model: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.columns = df.columns.str.strip()
    # Extract sample id from gtpath, e.g. "generated\gt\sample_000497.krn"
    df["sample_id"] = df["gtpath"].str.extract(r"(sample_\d+)")
    df["model"] = model
    return df[["sample_id", "model", "OMR-NED (OMR-ED / total numsyms)"]].rename(
        columns={"OMR-NED (OMR-ED / total numsyms)": "OMR_NED"}
    )

frames = [load_eval(p, m) for m, p in CSV_FILES.items()]
evals = pd.concat(frames, ignore_index=True)
evals["OMR_NED"] = pd.to_numeric(evals["OMR_NED"], errors="coerce")
invalid_ned = evals["OMR_NED"].isna().sum()
if invalid_ned:
    print(f"\n  !  {invalid_ned} rows have non-numeric OMR_NED and will be skipped")

# -- 3. Join with page_texture ------------------------------------------------
evals["page_texture"] = evals["sample_id"].map(texture_map)
missing = evals["page_texture"].isna().sum()
if missing:
    print(f"\n  ?  {missing} rows could not be matched to a page_texture")
    # Show unmatched sample ids for debugging
    print("  Unmatched IDs (first 5):", evals[evals["page_texture"].isna()]["sample_id"].unique()[:5].tolist())

# -- 3b. Coverage table: processed/total and failed counts -------------------
texture_totals = pd.Series(texture_map).groupby(pd.Series(texture_map).values).size()

went_through = (
    evals.dropna(subset=["sample_id", "page_texture", "OMR_NED"])
    .groupby(["page_texture", "model"])["sample_id"]
    .nunique()
    .unstack("model")
    .fillna(0)
    .astype(int)
)

went_through = went_through.reindex(texture_totals.index, fill_value=0)

coverage = pd.DataFrame(index=went_through.index)
for model in went_through.columns:
    coverage[model] = [
        f"{ok}/{int(texture_totals.loc[texture])}"
        for texture, ok in went_through[model].items()
    ]

failed_counts = pd.DataFrame(index=went_through.index)
for model in went_through.columns:
    failed_counts[model] = [
        int(texture_totals.loc[texture]) - int(ok)
        for texture, ok in went_through[model].items()
    ]

print("\n-- Coverage by page_texture (went-through/total) -------------------------")
print(coverage.to_string())
print("\n-- Failed counts by page_texture ------------------------------------------")
print(failed_counts.to_string())

evals = evals.dropna(subset=["page_texture", "OMR_NED"])
print(f"\n  Matched rows: {len(evals)}")

# -- 4. Analyse: mean OMR-NED per (model � page_texture) ---------------------
pivot = (
    evals.groupby(["page_texture", "model"])["OMR_NED"]
    .agg(["mean", "median", "count"])
    .round(4)
    .rename(columns={"mean": "Mean NED", "median": "Median NED", "count": "n"})
)
print("\n-- Mean OMR-NED by page_texture and model ------------------------------")
print(pivot.to_string())

# Wide table: models as columns, sorted by overall difficulty
wide = (
    evals.groupby(["page_texture", "model"])["OMR_NED"]
    .mean()
    .unstack("model")
    .round(4)
)
wide["Overall mean"] = wide.mean(axis=1)
wide = wide.sort_values("Overall mean")
print("\n-- Wide table (mean OMR-NED, lower = better) ---------------------------")
print(wide.to_string())

# Sample counts per texture
counts = evals.groupby("page_texture")["sample_id"].nunique().rename("n_samples")
print("\n-- Sample counts per page_texture --------------------------------------")
print(counts.to_string())

# -- 5. Analyse on shared coverage only -------------------------------------
common_sample_ids = (
    evals.groupby("sample_id")["model"]
    .nunique()
    .loc[lambda series: series == len(CSV_FILES)]
    .index
)
common_evals = evals[evals["sample_id"].isin(common_sample_ids)]

print(
    "\n-- Shared coverage across all models -----------------------------------"
)
print(f"  Common recognized samples: {len(common_sample_ids)}")

common_wide = (
    common_evals.groupby(["page_texture", "model"])["OMR_NED"]
    .mean()
    .unstack("model")
    .round(4)
)
common_wide["Overall mean"] = common_wide.mean(axis=1)
common_wide = common_wide.sort_values("Overall mean")
print(
    "\n-- Wide table (mean OMR-NED, lower = better) | shared samples only ----"
)
print(common_wide.to_string())

common_counts = (
    common_evals.groupby("page_texture")["sample_id"]
    .nunique()
    .rename("n_samples")
)
print("\n-- Sample counts per page_texture | shared samples only ---------------")
print(common_counts.to_string())
