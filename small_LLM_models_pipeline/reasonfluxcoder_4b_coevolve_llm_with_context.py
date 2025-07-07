import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sys
import subprocess
import importlib.util
import time

LOG_FILE = "translation_and_test_log.txt"

def log_write(msg):
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")
    print(msg)
# --------- 1. MANUAL EXAMPLES WITH PROBLEM STATEMENTS ---------
examples = [
    {
        "id": "e1.py",
        "problem": [
    "# Problem: Implement Linear Regression\n",
    "\n",
    "### Problem Statement\n",
    "Your task is to implement a **Linear Regression** model using PyTorch. The model should predict a continuous target variable based on a given set of input features.\n",
    "\n",
    "### Requirements\n",
    "1. **Model Definition**:\n",
    "   - Implement a class `LinearRegressionModel` with:\n",
    "     - A single linear layer mapping input features to the target variable.\n",
    "2. **Forward Method**:\n",
    "   - Implement the `forward` method to compute predictions given input data."
   ],
        "code": '''
import torch
import torch.nn as nn
import torch.optim as optim

# Generate synthetic data
torch.manual_seed(42)
X = torch.rand(100, 1) * 10  # 100 data points between 0 and 10
y = 2 * X + 3 + torch.randn(100, 1)  # Linear relationship with noise

# Define the Linear Regression Model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # Single input and single output

    def forward(self, x):
        return self.linear(x)

# Initialize the model, loss function, and optimizer
model = LinearRegressionModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
epochs = 1000
for epoch in range(epochs):
    # Forward pass
    predictions = model(X)
    loss = criterion(predictions, y)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Log progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# Display the learned parameters
[w, b] = model.linear.parameters()
print(f"Learned weight: {w.item():.4f}, Learned bias: {b.item():.4f}")

# Testing on new data
X_test = torch.tensor([[4.0], [7.0]])
with torch.no_grad():
    predictions = model(X_test)
    print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")
'''
    },
    {
        "id": "example2.py",
        "problem": [
    "## Problem: Quantize Your Language Model\n",
    "\n",
    "### Problem Statement\n",
    "Implement a **language model** using an LSTM and apply **dynamic quantization** to optimize it for inference. Dynamic quantization reduces the model size and enhances inference speed by quantizing the weights of the model.\n",
    "\n",
    "### Requirements\n",
    "\n",
    "1. **Define the Language Model**:\n",
    "   - **Purpose**: Build a simple language model that predicts the next token in a sequence.\n",
    "   - **Components**:\n",
    "     - **Embedding Layer**: Converts input tokens into dense vector representations.\n",
    "     - **LSTM Layer**: Processes the embedded sequence to capture temporal dependencies.\n",
    "     - **Fully Connected Layer**: Outputs predictions for the next token.\n",
    "     - **Softmax Layer**: Applies a probability distribution over the vocabulary for predictions.\n",
    "   - **Forward Pass**:\n",
    "     - Pass the input sequence through the embedding layer.\n",
    "     - Feed the embedded sequence into the LSTM.\n",
    "     - Use the final hidden state from the LSTM to make predictions via the fully connected layer.\n",
    "     - Apply the softmax function to obtain probabilities over the vocabulary.\n",
    "\n",
    "2. **Apply Dynamic Quantization**:\n",
    "   - Quantize the model dynamically\n",
    "   - Evaluate the quantized model's performance compared to the original model."
   ],
        "code": '''
import torch
import torch.nn as nn
import torch.optim as optim

# Generate synthetic data
torch.manual_seed(42)
X = torch.rand(100, 1) * 10  # 100 data points between 0 and 10
y = 2 * X + 3 + torch.randn(100, 1)  # Linear relationship with noise

class LearnedSiLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, slope):
        # Save the input tensor and slope for backward computation
        ctx.save_for_backward(x)
        ctx.slope = slope
        return slope * x * torch.sigmoid(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve the input and slope saved in the forward pass
        x, = ctx.saved_tensors
        slope = ctx.slope
        sigmoid_x = torch.sigmoid(x)

        # Compute the gradient with respect to input (x)
        grad_input = grad_output * slope * (sigmoid_x + x * sigmoid_x * (1 - sigmoid_x))

        # Compute the gradient with respect to slope
        grad_slope = grad_output * x * sigmoid_x

        return grad_input, grad_slope


# Define the Linear Regression Model
class LinearRegressionModel(nn.Module):
    def __init__(self, slope=1):
        super().__init__()
        self.slope = nn.Parameter(torch.ones(1) * slope)

    def forward(self, x):
        # Use the custom LearnedSiLUFunction
        return LearnedSiLUFunction.apply(x, self.slope)

# Initialize the model, loss function, and optimizer
model = LinearRegressionModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
epochs = 1000
for epoch in range(epochs):
    # Forward pass
    predictions = model(X)
    loss = criterion(predictions, y)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Log progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# Display the learned parameters
[w, b] = model.linear.parameters()
print(f"Learned weight: {w.item():.4f}, Learned bias: {b.item():.4f}")

# Testing on new data
X_test = torch.tensor([[4.0], [7.0]])
with torch.no_grad():
    predictions = model(X_test)
    print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")
'''
    }
]

# --------- 2. LLM PROMPT TEMPLATES ---------
def make_translation_prompt(problem, pytorch_code):
    return f"""You are an expert Python and JAX developer.
Problem statement: {problem}

The following is a PyTorch function:

{pytorch_code}

Translate this function to JAX, using only valid code. No markdown or extra explanation.
JAX version:
"""

def make_test_prompt(problem, jax_code):
    return f"""Given the following problem statement:
{problem}

And the following JAX function:

{jax_code}

Write 2-3 PyTest unit tests to check the correctness of the function, using only standard Python and JAX/numpy. No markdown or explanations.
"""

# --------- 3. LLM SETUP ---------
model_name="Gen-Verse/ReasonFlux-Coder-4B"
print(f"Loading model {model_name} ...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
).eval()

def generate_code(prompt, tokenizer, model, max_new_tokens=2048):
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return full_output[len(prompt):].strip().replace("```", "")
    
# --------- 4. TRANSLATE, TEST GENERATE, AND VERIFY LOOP ---------
with open(LOG_FILE, "w") as f:
    f.write(f"--- Translation and Test Log ({time.ctime()}) ---\n")

for ex in examples:
    base = ex['id'].replace('.py', '')

    # Translate
    translation_prompt = make_translation_prompt(ex["problem"], ex["code"])
    log_write(f"\n--- Translating {ex['id']} ---")
    log_write("Prompt to LLM for translation:\n" + translation_prompt)
    jax_code = generate_code(translation_prompt, tokenizer, model)
    log_write(f"Generated JAX code for {ex['id']}:\n{jax_code}")

    # Write JAX code to file
    jax_path = f"{base}_jax.py"
    with open(jax_path, "w") as f:
        f.write(jax_code)
    log_write(f"Saved JAX code to {jax_path}")

    # Generate test cases
    test_prompt = make_test_prompt(ex["problem"], jax_code)
    log_write(f"\n--- Generating unit tests for {base} ---")
    log_write("Prompt to LLM for test generation:\n" + test_prompt)
    test_code = generate_code(test_prompt, tokenizer, model)
    log_write(f"Generated PyTest code for {base}:\n{test_code}")

    # Write test code to file
    test_path = f"{base}_test.py"
    with open(test_path, "w") as f:
        f.write(test_code)
    log_write(f"Saved test code to {test_path}")

    # Compose test runner for pytest
    runner_code = f"""
import pytest
import numpy as np
import jax
import jax.numpy as jnp

from {base}_jax import *

{test_code}
"""
    runner_path = f"{base}_runner.py"
    with open(runner_path, "w") as f:
        f.write(runner_code)

    # Attempt to run Pytest up to 3 times (regenerate if fail)
    for attempt in range(1, 4):
        log_write(f"\n--- Running tests for {base} (attempt {attempt}) ---")
        result = subprocess.run([sys.executable, "-m", "pytest", runner_path], capture_output=True, text=True)
        log_write("Pytest output:\n" + result.stdout)
        if "FAILED" not in result.stdout:
            log_write(f"✅ {base}: Tests passed!")
            break
        else:
            log_write(f"❌ {base}: Tests failed, regenerating JAX code and trying again...")
            jax_code = generate_code(translation_prompt, tokenizer, model)
            with open(jax_path, "w") as f:
                f.write(jax_code)
            log_write("Regenerated JAX code:\n" + jax_code)
    else:
        log_write(f"❌ {base}: All attempts failed. Please review translation and tests manually.")

log_write("\nPipeline completed.")




# import os
# import json
# import re
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch

# MODEL_NAME = "Gen-Verse/ReasonFlux-Coder-4B"  
# INPUT_EXAMPLES_DIR = os.getenv('OUTPUT_FOLDER_PATH', 'reasonfluxcoder_4b_jax_translated_code')
# MAX_FILES = 10   # start small for testing!
# OUTPUT_DIR_BASE = "translated_code_OUTPUT"

# def load_pytorch_files(examples_dir, max_files=10):
#     code_files = []
#     count = 0
#     for root, _, files in os.walk(examples_dir):
#         for fname in files:
#             if fname.endswith(".py"):
#                 full_path = os.path.join(root, fname)
#                 with open(full_path, "r") as f:
#                     code = f.read()
#                 # Try to extract problem description (if present as top comment/docstring)
#                 problem = ""
#                 m = re.search(r'"""(.+?)"""', code, re.DOTALL)
#                 if m:
#                     problem = m.group(1).strip()
#                 code_files.append({
#                     "id": fname,
#                     "path": full_path,
#                     "problem": problem,
#                     "code": code
#                 })
#                 count += 1
#                 if count >= max_files:
#                     return code_files
#     return code_files

# def create_jax_prompt(code, problem="", reasoning_mode=True):
#     if reasoning_mode:
#         instruction = (
#             "You are a helpful AI that reasons step-by-step to translate Python code written using PyTorch to equivalent JAX code. "
#             "First, analyze step by step. Then, write the JAX version. Output only code (no explanation or markdown).\n" \
#             "Make sure you also write main functions for all of the JAX code. After this, write a unit test for each translated jax program," \
#             "then use this and the problem description as ground truth to regenerate jax code if necessary. "
#         )
#     else:
#         instruction = (
#             "Translate the following Python code using PyTorch to JAX. Output only valid JAX code. No explanation or markdown.\n"
#         )
#     if problem:
#         prompt = f"Problem description:\n{problem}\n\nPyTorch code:\n{code}\n\nJAX version:"
#     else:
#         prompt = f"{code}\n\nJAX version:"
#     return instruction + prompt

# def clean_llm_output(raw_output):
#     blocks = re.findall(r"```[a-zA-Z0-9]*\n(.*?)```", raw_output, re.DOTALL)
#     if blocks:
#         return blocks[0].strip()
#     return raw_output.replace("```", "").strip()

# def translate_code(code, tokenizer, model, problem="", reasoning_mode=False):
#     prompt = create_jax_prompt(code, problem, reasoning_mode)
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=2048,
#         temperature=0.0,
#         do_sample=False,
#         pad_token_id=tokenizer.eos_token_id
#     )
#     full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     torch.cuda.empty_cache()
#     # Remove prompt from the start if present
#     if full_output.startswith(prompt):
#         full_output = full_output[len(prompt):].strip()
#     return clean_llm_output(full_output)

# def save_translated_codes(examples, output_dir):
#     os.makedirs(output_dir, exist_ok=True)
#     for ex in examples:
#         base = os.path.splitext(ex["id"])[0]
#         filename = os.path.join(output_dir, f"{base}_jax.py")
#         with open(filename, "w") as f:
#             f.write(ex["translated_code"])

# def main():
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
#     model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)

#     pytorch_examples = load_pytorch_files(INPUT_EXAMPLES_DIR, max_files=MAX_FILES)

#     for mode_name, reasoning in [("no_reasoning", False), ("with_reasoning", True)]:
#         print(f"--- Running {mode_name.replace('_',' ').title()} Mode ---")
#         translated_examples = []
#         output_dir = os.path.join(OUTPUT_DIR_BASE, f"coevolve_ReasonFluxCoder4B_jax_{mode_name}_translated")
#         for ex in pytorch_examples:
#             print(f"Translating {ex['id']}...")
#             jax_code = translate_code(ex["code"], tokenizer, model, ex.get("problem", ""), reasoning_mode=reasoning)
#             translated_examples.append({
#                 "id": ex["id"],
#                 "problem": ex.get("problem", ""),
#                 "code": ex["code"],
#                 "translated_code": jax_code
#             })
#         save_translated_codes(translated_examples, output_dir)
#         # Optionally, save as JSON for further analysis
#         with open(os.path.join(output_dir, "all_translations.json"), "w") as f:
#             json.dump(translated_examples, f, indent=2)
#         print(f"✅ Translations saved to {output_dir}")

# if __name__ == "__main__":
#     main()
