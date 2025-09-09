from datasets import load_dataset
from huggingface_hub import login
import json


#General Thought Archive dataset: https://huggingface.co/datasets/RJT1990/GeneralThoughtArchive

# ---- Step 2: Load dataset ----
ds = load_dataset("RJT1990/GeneralThoughtArchive", split="train")

# ---- Optional: Test on first 5 rows ----
# ds = ds.select(range(5))

# ---- Step 3: Conversion function ----
def convert_chat(row):
    return {
        "messages": [
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row["reference_answer"]}
        ]
    }

# ---- Step 4: Apply conversion ----
chat_data = ds.map(convert_chat, remove_columns=ds.column_names)

# ---- Step 5: Save JSONL ----
output_file = "data_generated/train_chat.jsonl"
with open(output_file, "w", encoding="utf-8") as f:
    for ex in chat_data:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"✅ Saved {output_file} ({len(chat_data)} rows)")


#medical r1 distill data: https://huggingface.co/datasets/FreedomIntelligence/Medical-R1-Distill-Data

ds2 = load_dataset("FreedomIntelligence/Medical-R1-Distill-Data", split="train")

def convert_chat2(row):
    return {
        "messages": [
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row["response (content)"]}
        ]
    }

chat_data2 = ds2.map(convert_chat2, remove_columns=ds2 .column_names)

output_file2 = "data_generated/train_chat2.jsonl"
with open(output_file2, "w", encoding="utf-8") as f:
    for ex in chat_data2:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"✅ Saved {output_file2} ({len(chat_data2)} rows)")