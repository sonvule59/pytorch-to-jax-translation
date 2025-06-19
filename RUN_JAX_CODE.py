import os
import subprocess
import time
import json

INPUT_DIR = "./llama-3.1_Nemotron_Nano_4B-v1.1_jax_with_reasoning_translated"
dir_name = os.path.basename(os.path.normpath(INPUT_DIR))
LOG_JSON = f"{dir_name}_run_log.json"
ERROR_LOG = f"{dir_name}_run_errors.txt"

# LOG_JSON = "jax_run_log.json"
# ERROR_LOG = "jax_run_errors.txt"
TIMEOUT = 30  # seconds

results = []
error_messages = []

os.makedirs(INPUT_DIR, exist_ok=True)
files = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".py"))

for fname in files:
    path = os.path.join(INPUT_DIR, fname)
    print(f"▶️ Running: {fname}")
    result = {
        "file": fname,
        "run_success": False,
        "run_stdout": "",
        "run_stderr": "",
        "run_time_seconds": None
    }

    try:
        start = time.time()
        proc = subprocess.run(
            ["python3", path],
            capture_output=True,
            text=True,
            timeout=TIMEOUT
        )
        end = time.time()

        result["run_stdout"] = proc.stdout
        result["run_stderr"] = proc.stderr
        result["run_success"] = proc.returncode == 0
        result["run_time_seconds"] = round(end - start, 4)

        if proc.returncode != 0:
            error_messages.append(f"🛑 Runtime error in: {fname}\n{proc.stderr}\n")

    except subprocess.TimeoutExpired:
        result["run_stderr"] = "Timeout"
        result["run_time_seconds"] = TIMEOUT
        error_messages.append(f"⏰ Timeout for: {fname}\n")

    except Exception as e:
        result["run_stderr"] = str(e)
        error_messages.append(f"❌ Exception for: {fname}\n{str(e)}\n")

    results.append(result)

# Save results
with open(LOG_JSON, "w") as f:
    json.dump(results, f, indent=2)

with open(ERROR_LOG, "w") as f:
    f.writelines(error_messages)

print(f"✅ Done. Logs saved to '{LOG_JSON}' and '{ERROR_LOG}'")
