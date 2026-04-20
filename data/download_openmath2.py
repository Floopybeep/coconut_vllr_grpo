"""
Download nvidia/OpenMathInstruct-2 and filter to 'augmented_math' problem_source,
then save in the same format as svamp_test.json.
"""

import json
from datasets import load_dataset

print("Loading nvidia/OpenMathInstruct-2 ...")
ds = load_dataset("nvidia/OpenMathInstruct-2", split="train")

print(f"Total examples: {len(ds)}")

# filtered = ds.filter(lambda x: x["problem_source"] == "augmented_math")
filtered = ds.filter(lambda x: x["problem_source"] == "augmented_gsm8k")
print(f"After filtering to 'augmented_gsm8k': {len(filtered)}")

output = []
for idx, row in enumerate(filtered):
    if idx == 1000:
        break
    output.append({
        "question": row["problem"],
        "steps": ["."],
        "answer": str(row["expected_answer"]),
    })

out_path = "openmath2_augmented_gsm8k_1k.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=4)

print(f"Saved {len(output)} examples to {out_path}")
