"""
Download nvidia/OpenMathInstruct-2 and filter to 'augmented_math' problem_source,
then save in the same format as svamp_test.json.
"""

import json
from datasets import load_dataset

print("Loading HuggingFaceH4/MATH-500 ...")
ds = load_dataset("HuggingFaceH4/MATH-500", split="test")

print(f"Total examples: {len(ds)}")

# filtered = ds.filter(lambda x: x["problem_source"] == "augmented_math")
# filtered = ds.filter(lambda x: x["problem_source"] == "augmented_gsm8k")
# print(f"After filtering to 'augmented_gsm8k': {len(filtered)}")

output = []
for idx, row in enumerate(ds):
    # if idx == 1000:
    #     break
    output.append({
        "question": row["problem"],
        "steps": ["."],
        "answer": str(row["answer"]),
        "level": row["level"],
        "subject": row["subject"]
    })

out_path = "/data/math500_test.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=4)

print(f"Saved {len(output)} examples to {out_path}")
