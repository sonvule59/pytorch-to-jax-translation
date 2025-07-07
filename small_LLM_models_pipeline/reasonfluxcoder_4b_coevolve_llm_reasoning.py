import os
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from dotenv import load_dotenv

# Load .env
load_dotenv()
INPUT_DIR = os.getenv('INPUT_FOLDER_PATH')  
OUTPUT_DIR = os.getenv('OUTPUT_FOLDER_PATH', 'reasonfluxcoder_4b_jax_translated_code')
MODEL_NAME = "Gen-Verse/ReasonFlux-Coder-4B"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_model(model_name=MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    return tokenizer, model

def load_code_files(folder_path, max_files=10):
    files = []
    for i, filename in enumerate(sorted(os.listdir(folder_path))):
        if filename.endswith(".py"):
            path = os.path.join(folder_path, filename)
            with open(path, "r") as f:
                code = f.read()
                files.append({"id": filename, "code": code})
            if i + 1 >= max_files:
                break
    return files

def create_prompt(code, reasoning=True):
    if reasoning:
        prompt = (
            "You are an expert code translator. Given the following PyTorch code, "
            "reason step-by-step about its function, and then output an equivalent JAX implementation. " \
            "Make sure you also write main functions for all of the JAX code"
            "Output ONLY the JAX code (no markdown, no explanations).\n\n"
            "PyTorch code:\n"
            f"{code}\n\n"
            "JAX code:"
        )
    else:
        prompt = (
            "Translate the following PyTorch code to JAX. Output only valid JAX code, no explanations.\n\n"
            f"{code}\n\nJAX code:"
        )
    return prompt

def clean_code(raw):
    # Extract code block if present
    code_block = re.findall(r"```(?:python|jax)?\n(.*?)```", raw, re.DOTALL)
    if code_block:
        return code_block[0].strip()
    return raw.replace("```", "").strip()

def translate(code, tokenizer, model, reasoning=True):
    prompt = create_prompt(code, reasoning=reasoning)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=16384).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    torch.cuda.empty_cache()
    # Remove prompt from output
    translated = full_output[len(prompt):].strip()
    return clean_code(translated)

def save_jax_files(examples, outdir):
    os.makedirs(outdir, exist_ok=True)
    for ex in examples:
        base = os.path.splitext(ex["id"])[0]
        jax_path = os.path.join(outdir, f"{base}_jax.py")
        with open(jax_path, "w") as f:
            f.write(ex["translated_code"])

if __name__ == "__main__":
    print(f"📂 Loading PyTorch files from: {INPUT_DIR}")
    examples = load_code_files(INPUT_DIR)
    print(f"🚀 Loading ReasonFlux-Coder-4B model...")
    tokenizer, model = load_model()

    results = []
    for i, ex in enumerate(examples):
        print(f"🔹 [{i+1}/{len(examples)}] Translating {ex['id']} ...")
        try:
            jax_code = translate(ex["code"], tokenizer, model, reasoning=True)
        except Exception as e:
            jax_code = f"# TRANSLATION ERROR: {e}"
        results.append({
            "id": ex["id"],
            "pytorch_code": ex["code"][:500] + ("..." if len(ex["code"]) > 500 else ""),
            "translated_code": jax_code
        })
    save_jax_files(results, OUTPUT_DIR)
    # Save summary as JSON
    with open(os.path.join(OUTPUT_DIR, "results_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"✅ Done. Translated files in '{OUTPUT_DIR}'.")
