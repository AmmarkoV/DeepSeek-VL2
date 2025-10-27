#pip install -U transformers accelerate bitsandbytes

import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

MODEL_ID = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"

def load_model():
    # quantization config (example 4-bit)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    print(f"Loading tokenizer for {MODEL_ID} …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    print(f"Loading model {MODEL_ID} …")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model.config.use_cache = True
    return tokenizer, model

tokenizer, model = load_model()

def inference_fn(prompt: str, temperature: float = 0.6, max_tokens: int = 512):
    # Ensure temperature is > 0
    temperature = max(0.01, float(temperature))

    system_prompt = "This is DeepSeek-R1-0528-Qwen3-8B responding.\n"
    full_prompt = system_prompt + prompt

    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    output = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    return output[len(full_prompt):].strip()


iface = gr.Interface(
    fn=inference_fn,
    inputs=[
        gr.Textbox(lines=6, label="Prompt"),
        gr.Slider(minimum=0.01, maximum=1.0, value=0.6, label="Temperature"),
        gr.Slider(minimum=16, maximum=2048, value=512, step=16, label="Max new tokens")
    ],
    outputs=[gr.Textbox(lines=6, label="Model response")],
    title="DeepSeek-R1-0528-Qwen3-8B Chat",
    description="Chat with DeepSeek-R1-0528-Qwen3-8B model."
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=8083, share=False)

