from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from dotenv import load_dotenv

load_dotenv()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Load the model and tokenizer
model_name = "Gen-Verse/ReasonFlux-Coder-4B"
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
        max_new_tokens=512,
        do_sample=False,  # deterministic
        temperature=0.0,
        pad_token_id=tokenizer.eos_token_id
    )
    

# Decode and print result
generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_code)
