"""
Merges attention benchmark results (data/attention_benchmark/*.csv) into a
single csv and markdown.
"""

import pandas as pd

dir = "data/attention_benchmark"
baseline = "baseline"
compile = "compile"
compile_float32 = "compile_float32_high_precision"


# Load the three CSVs
df1 = pd.read_csv(f"{dir}/{baseline}.csv")
df2 = pd.read_csv(f"{dir}/{compile}.csv")
df3 = pd.read_csv(f"{dir}/{compile_float32}.csv")

# Rename relevant columns to distinguish them
df1_renamed = df1.rename(
    columns={"forward time": "baseline forward", "backward time": "baseline backward"}
)

df2_renamed = df2.rename(
    columns={"forward time": "compile forward", "backward time": "compile backward"}
)

df3_renamed = df3.rename(
    columns={
        "forward time": "compile w/ float32 forward",
        "backward time": "compile w/ float32 backward",
    }
)

# Merge on d_model and seq_len
merged = df1_renamed.merge(
    df2_renamed, on=["d_model", "seq_len"], suffixes=("", "_drop")
)
merged = merged.merge(df3_renamed, on=["d_model", "seq_len"], suffixes=("", "_drop"))

# Drop any duplicate columns created by merging
merged = merged.loc[:, ~merged.columns.str.endswith("_drop")]

merged = merged[
    [
        "d_model",
        "seq_len",
        "baseline forward",
        "compile forward",
        "compile w/ float32 forward",
        "baseline backward",
        "compile backward",
        "compile w/ float32 backward",
    ]
]

# Save merged output to CSV
merged.to_csv(f"{dir}/merged_output.csv", index=False)

# Optional: print as markdown table
with open(f"{dir}/merged_output.md", "w") as f:
    f.write(merged.to_markdown(index=False))
