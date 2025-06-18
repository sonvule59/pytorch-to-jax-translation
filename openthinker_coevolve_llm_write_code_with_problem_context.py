"""This program writes a program based on the prompt's problem description FROM the user.
"""
import os
import dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from dotenv import load_dotenv

load_dotenv()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Load the model and tokenizer
model_name = "open-thoughts/OpenThinker3-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16).eval()

# Define the prompt
prompt = """You are an expert programmer. Write a Python function to check if a number is prime.
The function should return True if the number is prime, and False otherwise.
Then generate unit tests based on the output. Just write code. No explanation or comments
"""
# Tokenize input
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate output
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=False,  # deterministic
        temperature=0.0,
        pad_token_id=tokenizer.eos_token_id
    )
    torch.cuda.empty_cache()


# Decode result
generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Define output folder and file
output_folder = "openthinker_generated_code"
output_file = os.path.join(output_folder, "prime_checker.py")

# Create folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Write to file
with open(output_file, "w") as f:
    f.write(generated_code)

# # Decode and print result
# generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print(generated_code)
