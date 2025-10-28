from transformers import AutoTokenizer
import os

model_name = "m-a-p/YuE-s1-7B-anneal-en-cot"
save_dir = "./tokenizer"
os.makedirs(save_dir, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(save_dir)
print(f"Tokenizer saved to {save_dir}")

