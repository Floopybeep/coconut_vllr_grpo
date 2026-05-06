"""
Download nvidia/OpenMathInstruct-2 and filter to 'augmented_math' problem_source,
then save in the same format as svamp_test.json.
"""

import json
from datasets import load_dataset

print("Loading reasoning-machines/gsm-hard ...")
ds = load_dataset("reasoning-machines/gsm-hard", split="test")

print(f"Total examples: {len(ds)}")

# filtered = ds.filter(lambda x: x["problem_source"] == "augmented_math")
# filtered = ds.filter(lambda x: x["problem_source"] == "augmented_gsm8k")
# print(f"After filtering to 'augmented_gsm8k': {len(filtered)}")

output = []
for idx, row in enumerate(ds):
    # if idx == 1000:
    #     break
    output.append({
        "question": row["input"],
        "steps": ["."],
        "answer": str(row["target"]),
        # "level": row["level"],
        # "subject": row["subject"]
    })

out_path = "/data/gsm_hard_test.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=4)

print(f"Saved {len(output)} examples to {out_path}")
