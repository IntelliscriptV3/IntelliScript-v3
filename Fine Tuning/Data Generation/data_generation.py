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



# UCSC-VLAA/m23k-tokenized : https://huggingface.co/datasets/UCSC-VLAA/m23k-tokenized
ds3 = load_dataset("UCSC-VLAA/m23k-tokenized",split="train")

def convert_chat3(row):
    return {
        "messages": [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["answer_string"]}
        ]
    }

chat_data3 = ds3.map(convert_chat3, remove_columns=ds3 .column_names)

output_file3 = "data_generated/train_chat3.jsonl"
with open(output_file3, "w", encoding="utf-8") as f:
    for ex in chat_data3:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"✅ Saved {output_file3} ({len(chat_data3)} rows)")

# MedMcQA: https://huggingface.co/datasets/medmcqa/medmcqa
ds_mcq = load_dataset("openlifescienceai/medmcqa",split="train")

# ---- Step 2: Conversion function ----
def convert_mcq(row):
    # Format the options
    options_text = f"A) {row['opa']}\nB) {row['opb']}\nC) {row['opc']}\nD) {row['opd']}"
    
    # Build user prompt
    user_prompt = f"Question: {row['question']}\nOptions:\n{options_text}\nChoose the correct option."
    
    # Build assistant answer (correct option + explanation)
    assistant_answer = f"The correct answer is {row['cop']}.\nExplanation: {row['exp']}"
    
    return {
        "messages": [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_answer}
        ]
    }

# ---- Step 3: Apply conversion ----
chat_data_mcq = ds_mcq.map(convert_mcq, remove_columns=ds_mcq.column_names)

# ---- Step 4: Save to JSONL ----
output_file_mcq = "data_generated/train_chat_mcq.jsonl"
with open(output_file_mcq, "w", encoding="utf-8") as f:
    for ex in chat_data_mcq:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"✅ Saved {output_file_mcq} ({len(chat_data_mcq)} rows)")


# PubmedQA: qiaojin/PubMedQA"
ds4 = load_dataset("qiaojin/PubMedQA", "pqa_artificial",split="train")

def convert_chat4(row):
    return {
        "messages": [
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row["long_answer"]}
        ]
    }

chat_data4 = ds4.map(convert_chat4, remove_columns=ds4.column_names)

output_file4 = "data_generated/train_chat4.jsonl"
with open(output_file4, "w", encoding="utf-8") as f:
    for ex in chat_data4:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"✅ Saved {output_file4} ({len(chat_data4)} rows)")

# MedReason "UCSC-VLAA/MedReason"

# ---- Step 1: Load dataset ----
ds_mcq2 = load_dataset("UCSC-VLAA/MedReason", split="train")

# ---- Step 2: Conversion function ----
def convert_mcq2(row):
    # User prompt: question + options
    user_prompt = f"Question: {row['question']}\nOptions:\n{row['options']}\nChoose the correct option."
    
    # Assistant reply: correct answer
    assistant_answer = f"The correct answer is: {row['answer']}."
    
    return {
        "messages": [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_answer}
        ]
    }

# ---- Step 3: Apply conversion ----
chat_data_mcq2 = ds_mcq2.map(convert_mcq2, remove_columns=ds_mcq2.column_names)

# ---- Step 4: Save as JSONL ----
output_file_mcq2 = "data_generated/train_chat_mcq2.jsonl"
with open(output_file_mcq2, "w", encoding="utf-8") as f:
    for ex in chat_data_mcq2:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"✅ Saved {output_file_mcq2} ({len(chat_data_mcq2)} rows)")



