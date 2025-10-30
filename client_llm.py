#!/bin/python3
# To get dependencies:
#     python3 -m pip install gradio_client

import json
import time
import sys
from gradio_client import Client

def remainingTimeString(seconds):
    if seconds < 60:
        return f"Remaining Time {seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds // 60
        return f"Remaining Time {minutes} minutes"
    elif seconds < 86400:
        hours = seconds // 3600
        return f"Remaining Time {hours} hours"
    else:
        days = seconds // 86400
        return f"Remaining Time {days} days"

# Default server and params
ip = "127.0.0.1"
port = "8083"
output_file = "output.json"

# Hyperparameters
temperature = 0.6
max_tokens = 512

prompts = []

# Parse arguments
i = 1
while i < len(sys.argv):
    arg = sys.argv[i]
    if arg == "--ip":
        ip = sys.argv[i + 1]
        i += 2
    elif arg == "--port":
        port = sys.argv[i + 1]
        i += 2
    elif arg in ("--output", "-o"):
        output_file = sys.argv[i + 1]
        i += 2
    elif arg == "--temperature":
        temperature = float(sys.argv[i + 1])
        i += 2
    elif arg == "--max_tokens":
        max_tokens = int(sys.argv[i + 1])
        i += 2
    else:
        prompts.append(arg)
        i += 1

if not prompts:
    prompts = ["Explain the benefits of quantization in large language models."]

client = Client(f"http://{ip}:{port}")

results = {}

for idx, prompt in enumerate(prompts):
    start = time.time()
    try:
        response = client.predict(
            prompt,         # prompt text
            temperature,    # temperature
            max_tokens,     # max new tokens
            api_name="/predict"  # default Gradio function
        )
    except Exception as e:
        print(f"❌ Error for prompt {idx}: {e}")
        break

    elapsed = time.time() - start
    remaining = (len(prompts) - idx - 1) * elapsed
    hz = 1 / (elapsed + 1e-6)

    print(f"✅ Processed {idx + 1}/{len(prompts)} | {hz:.2f} Hz | {remainingTimeString(remaining)}")
    print(f"Prompt: {prompt}\nResponse: {response}\n")

    results[prompt] = response

# Save results
print(f"\nSaving results to {output_file}")
with open(output_file, "w") as f:
    json.dump(results, f, indent=4)

