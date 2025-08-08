"""
Parses attention benchmark logs (data/attention_benchmark/*.log) into csv and markdowns.
"""

import re
import pandas as pd

# Input and output files
dir = "data/attention_benchmark"
name = "compile_float32_high_precision"
input_file = f"{dir}/{name}.log"
csv_output_file = f"{dir}/{name}.csv"
md_output_file = f"{dir}/{name}.md"

# Regex patterns
header_pattern = re.compile(r"== d_model: (\d+), seq_len: (\d+) ==")
forward_pattern = re.compile(r"Forward: ([\d.]+)s")
mem_forward_pattern = re.compile(r"CUDA memory usage after forward: ([\d.]+)MB")
backward_pattern = re.compile(r"Backward: ([\d.]+)s")

# Data collection
rows = []

with open(input_file, "r") as f:
    lines = f.readlines()

i = 0
while i < len(lines):
    header_match = header_pattern.match(lines[i].strip())
    if header_match:
        d_model = int(header_match.group(1))
        seq_len = int(header_match.group(2))

        forward_match = forward_pattern.match(lines[i + 1].strip())
        mem_forward_match = mem_forward_pattern.match(lines[i + 2].strip())
        backward_match = backward_pattern.match(lines[i + 3].strip())

        if forward_match and mem_forward_match and backward_match:
            forward_time = float(forward_match.group(1))
            mem_after_forward = float(mem_forward_match.group(1))
            backward_time = float(backward_match.group(1))

            rows.append(
                [d_model, seq_len, forward_time, mem_after_forward, backward_time]
            )
        i += 5
    else:
        i += 1

# Convert to DataFrame
df = pd.DataFrame(
    rows,
    columns=[
        "d_model",
        "seq_len",
        "forward time",
        "CUDA memory usage after forward",
        "backward time",
    ],
)

# Write CSV
df.to_csv(csv_output_file, index=False)

# Write Markdown
with open(md_output_file, "w") as f:
    f.write(df.to_markdown(index=False))

print(f"Wrote {len(df)} rows to '{csv_output_file}' and '{md_output_file}'")
