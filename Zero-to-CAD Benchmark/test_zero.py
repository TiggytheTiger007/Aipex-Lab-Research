import os
import sys
import re

# ---------------------------------------------------------
# 1. AUTO-PATCH C++ LIBRARY FIX (Must be at the very top)
# ---------------------------------------------------------
conda_prefix = os.environ.get("CONDA_PREFIX", "/home/aipexws2/anaconda3/envs/sid")
modern_lib = os.path.join(conda_prefix, "lib", "libstdc++.so.6")

# If the fix isn't applied yet, inject it and instantly restart the script
if os.environ.get("LD_PRELOAD") != modern_lib and os.path.exists(modern_lib):
    print("🔧 Auto-patching C++ library paths to prevent CadQuery crash...")
    os.environ["LD_PRELOAD"] = modern_lib
    
    # Restart the script seamlessly with the new environment variables
    os.execv(sys.executable, [sys.executable] + sys.argv)

# ---------------------------------------------------------
# 2. MODEL GENERATION
# ---------------------------------------------------------
# LOCK TO THE EMPTY GPU YOU FOUND (GPU 1)
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import cadquery as cq

def run_benchmark():
    model_id = "ADSKAILab/Zero-To-CAD-Qwen3-VL-2B"

    print(f"\n--- Loading Qwen3-VL Model: {model_id} ---")
    
    # Qwen3-VL uses a Processor, not just a Tokenizer
    processor = AutoProcessor.from_pretrained(model_id)
    
    # Use the specific Qwen3VL class
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16, # Better for your RTX 6000 Ada
        trust_remote_code=True
    )

    # Prompt from your spreadsheet
    user_prompt = "Generate a complete, executable CadQuery Python script for a cylindrical body with a centered hexagonal through-hole. Define explicit, human-readable parametric variables (such as cylinder_outer_radius, cylinder_height, and hex_hole_radius) to control the geometry, ensuring the resulting wall thickness is thin."
    
    # Formatting for Qwen3-VL
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are a helpful assistant."}, 
            ],
        },
        {
            "role": "user",
            "content": [
                # Keep using your highly structured user prompt here!
                {"type": "text", "text": f"Generate CadQuery Python code for: {user_prompt}"},
            ],
        }
    ]
    
    # Prepare inputs
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt", padding=True).to("cuda")

    print(f"--- Generating for Prompt: '{user_prompt}' ---\n")
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, 
            max_new_tokens=1024,
            do_sample=False
        )

    # Decode only the new parts (the response)
    generated_ids = [
        out[len(ins):] for ins, out in zip(inputs.input_ids, output_ids)
    ]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    print("="*40)
    print("CAD-GPT/ZERO-TO-CAD RESULT:")
    print("="*40)
    print(response)
    print("="*40)

    # ---------------------------------------------------------
    # 3. POST-PROCESSING & STL EXPORT LOGIC
    # ---------------------------------------------------------
    print("\n--- Attempting to Extract and Export STL ---")
    
    # Extract the raw python code from the LLM's markdown output
    match = re.search(r'```python\s*(.*?)\s*```', response, re.DOTALL)
    if match:
        cad_code = match.group(1)
        print("Successfully extracted Python code from markdown.")
    else:
        cad_code = response 
        print("No markdown formatting found. Attempting to run raw response.")

    local_namespace = {}
    
    try:
        # Execute the LLM-generated code safely
        exec(cad_code, globals(), local_namespace)
        
        cq_object = None
        
        # Locate the CadQuery object in the executed variables
        if 'result' in local_namespace and isinstance(local_namespace['result'], cq.Workplane):
            cq_object = local_namespace['result']
        else:
            for var_name, var_val in local_namespace.items():
                if isinstance(var_val, cq.Workplane):
                    cq_object = var_val
                    break
        
        # Export to STL
        if cq_object is not None:
            output_filename = "fine_tuned_prompt_1.stl"
            cq.exporters.export(cq_object, output_filename)
            print(f"✅ Success! 3D model exported to: {output_filename}")
        else:
            print("❌ Execution succeeded, but could not locate a CadQuery Workplane object to export.")
            
    except Exception as e:
        print(f"❌ Failed to execute the generated CadQuery script. Error:\n{e}")

if __name__ == "__main__":
    run_benchmark()