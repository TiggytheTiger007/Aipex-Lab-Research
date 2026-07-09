import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Lock to your empty GPU

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# This script runs through a list of prompts, generating CAD code with and without reasoning.
# =====================================================================
PROMPTS_LIST = [
"A cylindrical sleeve with a central hexagonal through-hole resulting in a thin-walled profile.",
"A rectangular channel block featuring a central semi-cylindrical cutout that runs along its length.",
"A hollow half-pipe structure formed by a longitudinally cut semi-cylinder.",
"A flat mounting plate with a hexagonal profile.",
"A teardrop-shaped mechanical linkage featuring a large circular bore at the broad end and a smaller through-hole at the narrow end.",
"A cylindrical spacer containing a centered square through-hole extending along its entire vertical axis.",
"A base component featuring four vertical cylindrical mounting posts distributed asymmetrically.",
"A thin trapezoidal prism functioning as a structural plate.",
"A single component consisting of three parallel, evenly spaced vertical rectangular fins joined at a common base.",
"A hollow cylindrical retaining ring with a vertical cutaway creating a C-shaped profile."
]

def load_model():
    model_id = "ADSKAILab/Zero-To-CAD-Qwen3-VL-2B"
    print("--- Loading Model ---")
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    return model, processor

def generate_cad(model, processor, system_instruction, user_prompt):
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": [
            {"type": "text", "text": user_prompt}
        ]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt", padding=True).to("cuda")
    
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        
    generated_ids = [out[len(ins):] for ins, out in zip(inputs.input_ids, output_ids)]
    return processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

def main():
    model, processor = load_model()
    output_file = "/home/aipexws2/Sid/Zero-to-CAD Benchmark/benchmark_results.txt"
    
    with open(output_file, "w") as f:
        f.write("=== ZERO-TO-CAD AUTOMATED BENCHMARK RESULTS ===\n\n")
        
    for idx, raw_prompt in enumerate(PROMPTS_LIST, start=1): # Change start number to match your spreadsheet row
        print(f"\n🚀 Processing Prompt {idx}/{len(PROMPTS_LIST)}...")
        
        # Run Baseline (No Reasoning)
        sys_direct = "You are a CAD code assistant. Generate clean CadQuery Python code."
        prompt_direct = f"Generate CadQuery Python code for: {raw_prompt}"
        res_direct = generate_cad(model, processor, sys_direct, prompt_direct)
        
        # Run with Reasoning Asked
        prompt_reason = f"Let's think step by step. Generate CadQuery Python code for: {raw_prompt}"
        res_reason = generate_cad(model, processor, sys_direct, prompt_reason)
        
        # Save results to file
        with open(output_file, "a") as f:
            f.write(f"========================================\n")
            f.write(f"PROMPT #{idx}: {raw_prompt}\n")
            f.write(f"========================================\n\n")
            f.write(f"--- SHEET 1: DIRECT OUTPUT ---\n")
            f.write(f"{res_direct}\n\n")
            f.write(f"--- SHEET 2: REASONING ASKED OUTPUT ---\n")
            f.write(f"{res_reason}\n\n")
            f.write(f"========================================\n\n\n")
            
    print(f"\n Results saved to: {output_file}")

if __name__ == "__main__":
    main()