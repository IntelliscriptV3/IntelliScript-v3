from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

# ---- Step 1: Login to Hugging Face ----

model_name = "Qwen/Qwen3-8B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Save locally
save_dir = "local_model"
tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)
