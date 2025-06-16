import os
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from dotenv import load_dotenv

load_dotenv()
INPUT_FILES = os.getenv('INPUT_FOLDER_PATH')
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ----- LOAD MODEL -----
def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    return tokenizer, model

# ----- LOAD FILES -----
def load_code_files(folder_path, max_files=5):
    code_examples = []
    count = 0
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".py"):
            full_path = os.path.join(folder_path, filename)
            with open(full_path, "r") as f:
                code = f.read()
                code_examples.append({
                    "id": filename,
                    "language": "python",
                    "code": code
                })
            count += 1
            if count >= max_files:
                break
    return code_examples

# ----- PROMPT CREATION -----
def create_jax_prompt(code, reasoning_mode=False):
    if reasoning_mode:
        instruction = (
            "You are a helpful AI that reasons step-by-step to translate Python code written using PyTorch to equivalent JAX code. "
            "First analyze what the code is doing, then translate the full JAX version. No explanation or markdown. Output only code.\n\n"
        )
    else:
        instruction = (
            "Translate the following Python code using PyTorch to JAX. Output only valid JAX code. No explanation or markdown.\n\n"
        )

    max_code_length = 14000
    if len(code) > max_code_length:
        code = code[:max_code_length]

    return instruction + code + "\n\nJAX version:"

# ----- CLEAN OUTPUT -----
def clean_translated_code(raw_output):
    blocks = re.findall(r"```[a-zA-Z0-9]*\n(.*?)```", raw_output, re.DOTALL)
    if blocks:
        return blocks[0].strip()
    return raw_output.replace("```", "").strip()

# ----- TRANSLATION -----
def translate_code_to_jax(code, tokenizer, model, reasoning_mode=False):
    prompt = create_jax_prompt(code, reasoning_mode)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.0,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    torch.cuda.empty_cache()
    return clean_translated_code(full_output[len(prompt):].strip())

# ----- SAVE FILES -----
def save_jax_files(examples, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for ex in examples:
        base = os.path.splitext(ex["id"])[0]
        filename = os.path.join(output_dir, f"{base}_jax.py")
        with open(filename, "w") as f:
            f.write(ex["translated_code"])

# ----- MAIN -----
if __name__ == "__main__":
    for mode_name, reasoning in [("no_reasoning", False), ("with_reasoning", True)]:
        print(f"===== Starting {mode_name.replace('_', ' ').title()} Mode =====")

        folder = INPUT_FILES
        print(f"📂 Loading files from: {folder}")
        code_examples = load_code_files(folder)

        print("🚀 Loading model...")
        tokenizer, model = load_model("nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1")  # Replace if needed

        print("🔁 Translating to JAX...")
        translated_examples = []
        for i, ex in enumerate(code_examples):
            print(f"🔹 Translating {ex['id']}")
            jax_code = translate_code_to_jax(ex["code"], tokenizer, model, reasoning_mode=reasoning)
            print("🔸 JAX preview:", "\n".join(jax_code.splitlines()[:5]), "\n...")
            translated_examples.append({
                "id": ex["id"],
                "language": ex["language"],
                "code": ex["code"],
                "translated_code": jax_code
            })

        save_dir = f"jax_{mode_name}_translated"
        save_jax_files(translated_examples, output_dir=save_dir)

        json_file = f"jax_{mode_name}_results_only_translation.json"
        print(f"💾 Saving output to '{json_file}'")
        with open(json_file, "w") as f:
            json.dump(translated_examples, f, indent=2)

        print(f"✅ Finished {mode_name.replace('_', ' ').title()} Mode\n")






# import os
# import dotenv
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch
# from dotenv import load_dotenv

# load_dotenv()
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# # Load the model and tokenizer
# model_name = "open-thoughts/OpenThinker3-7B"
# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16).eval()

# # Define the prompt
# prompt = """You are an expert programmer. Write a Python function to check if a number is prime."""
# The function should return True if the number is prime, and False otherwise.
# Then generate unit tests based on the output. Just write code. No explanation or comments
# """

# # Tokenize input
# inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# # Generate output
# with torch.no_grad():
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=1024,
#         do_sample=False,  # deterministic
#         temperature=0.7,
#         pad_token_id=tokenizer.eos_token_id
#     )
#     torch.cuda.empty_cache()
# # Decode and print result
# generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print(generated_code)
