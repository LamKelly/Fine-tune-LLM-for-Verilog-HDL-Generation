

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from peft import PeftModel

BASE_MODEL = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"   # or your 8B model folder
FINETUNE_DIR = r"d:\ECE465Project\qlora_adapter_E3"        #r"d:\ECE465Project\qlora_adapter_E3"        # where trainer.model.save_pretrained() saved LoRA
#r"d:\ECE465Project\qlora_adapter_E3"        # where trainer.model.save_pretrained() saved LoRA

OUTPUT_MAX_LEN = 256
#%%
# ================================
# LOAD BASE MODEL (4-bit)
# ================================
print("Loading base tokenizer...")
base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if base_tokenizer.pad_token is None:
    base_tokenizer.pad_token = base_tokenizer.eos_token

print("Loading base 4-bit model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="cuda",
    trust_remote_code=True,
)



# ================================
# LOAD FINETUNED (LoRA) MODEL
# ================================
print("Loading finetuned LoRA adapter...")
ft_model = PeftModel.from_pretrained(
    base_model,          # attaches LoRA to the base model
    FINETUNE_DIR,
)
ft_model.eval()

# You MUST reuse the same tokenizer for base + finetuned
ft_tokenizer = base_tokenizer


# ================================
# GENERATION FUNCTION
# ================================
def generate_text(model_choice, user_prompt):

    # Format the prompt exactly like your training data
    formatted_prompt = f"Instruction:\n{user_prompt}\n\nResponse:\n"

    # Choose model + tokenizer based on user selection
    if model_choice == "Original Model":
        model = base_model
        tokenizer = base_tokenizer
    else:
        model = ft_model
        tokenizer = ft_tokenizer

    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    # Generate
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=OUTPUT_MAX_LEN,
            do_sample=True,
            temperature=0.1,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode
    text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Cut everything before "Response:" — optional but recommended
    if "Response:" in text:
        text = text.split("Response:", 1)[1].strip()

    return text

# ================================
# GRADIO UI
# ================================
interface = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Radio(
            ["Original Model", "Fine-Tuned Model"],
            label="Choose Model"
        ),
        gr.Textbox(
            lines=3,
            placeholder="Enter a Verilog-related prompt...",
            label="Prompt"
        )
    ],
    outputs=gr.Textbox(label="Generated Output"),
    title="OS-Verilog Copilot — Llama 3 QLoRA Verilog Generator",
    description="Compare the base model vs your fine-tuned Verilog LLM."
)


if __name__ == "__main__":
    interface.launch(share=True)
