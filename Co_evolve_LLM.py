from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

# Load the model and tokenizer
model_name = "open-thoughts/OpenThinker3-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, trust_remote_code=True, torch_dtype=torch.float16
).eval()

# Define the prompt
prompt = """You are an expert programmer. Write a JAX function to check if a number is prime.
The function should return True if the number is prime, and False otherwise.
"""

# Tokenize input
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate output
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=False,  # deterministic
        temperature=0.0,
        pad_token_id=tokenizer.eos_token_id
    )

# Decode the result
generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Optional: strip the prompt part if needed
generated_code_only = generated_code[len(prompt):].strip()

# Output directory
output_dir = "generated_jax_code"
os.makedirs(output_dir, exist_ok=True)

# Save to file
output_file_path = os.path.join(output_dir, "is_prime_jax.py")
with open(output_file_path, "w") as f:
    f.write(generated_code_only)

print(f"✅ JAX code written to: {output_file_path}")
