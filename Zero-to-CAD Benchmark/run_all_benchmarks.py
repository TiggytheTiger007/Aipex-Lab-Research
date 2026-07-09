import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Lock to your empty GPU

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# This script runs through a list of prompts, generating CAD code with and without reasoning.
# =====================================================================
PROMPTS_LIST = [
    "The 3D shape is a cylinder and a hexagonal hole inside, which is smaller and makes the wall very thin.",
    "The 3D shape is a rectangular block with a semicylindrical cutout located at its center, forming a U-shaped channel.",
    "The 3D shape is a hollow, semi-cylindrical structure cut lengthwise, resembling a half-pipe.",
    "The 3D shape is a square hexagonal plate.",
    "The 3D shape is a teardrop-like piece with two circular holes. one large near the broader end and one small near the narrower end.",
    "The shape is a cylinder with a square hole centered at the top, extending from the top to the bottom.",
    "The shape is composed of four vertical cylinders, roughly the same size, unevenly distributed at the four corners.",
    "The 3D shape is a trapezoid thin prism.",
    "Three identical rectangular sheets placed vertically, arranged in parallel and evenly spaced.",
    "The shape is a hollow cylindrical band with a vertical sector removed, resembling an incomplete ring.",
    "The 3D shape is a hollow triangular prism. The walls are the same and have a smaller thickness.",
    "The image shows two identical parallel long slim pipes.",
    "The three-dimensional shape is an inverted T-shaped prism.",
    "The three-dimensional shape is a flattened cylinder.",
    "The 3D shape is a rectangular prism(cuboid).",
    "The 3D shape is a combination of a rectangular prism base and a vertically oriented half-cylinder on top.",
    "A flat rectangular plate. All four corners are rounded and there is a circular hole of the same diameter at each corner.",
    "The 3D shape consists of a small thin rectangular prism in the middle of the right side of a rectangular prism.",
    "The 3D shape is a hexagonal prism. The hollow center forms an open hexagonal cross-section.",
    "The 3D shape is a rectangular cuboid with rounded edges and corners.",
    "Two intersecting U-channels with vertical walls, covered by two rectangular plates at the intersection to form a sealed enclosure.",
    "A central square with four smaller squares at the corners and four semi-cylindrical shapes along its sides, between the smaller squares.",
    "A square block with a gear-like cutout, C-shaped extrusions on sides, and a cylindrical base with a smaller concentric protrusion.",
    "A disk base with ten evenly spaced holes, supporting a cylindrical assembly with semi-circular protrusions and hollow interior.",
    "A rectangluar block with opposite rectangular cutouts, four cylindrical tubes on one side, and a cross-shaped object between the tube sets.",
    "A square base with a central cutout and four corner holes, connected to a rectangular prism with V-shaped side and square top cutouts.",
    "A bowtie-shpaed plate with two cylindrical discs with holes on each side.",
    "A hollow rectangular prism containing a block with a central hole and side protrusions.",
    "A square frame containing a stair-step object with rounded edges.",
    "A rounded rectangular base with holes, two rounded cutouts, and a central ring.",
    "A rectangular block with ten parallel plates extending from one side.",
    "A square outer frame with a hollow, symmetrical structure suspended inside.",
    "The spring is a helical coil spring with specific dimensional properties. Its coil diameter is 72mm, and the pitch, which is the distance between adjacent coils, is 6mm. The free length of the spring, which is the length of the spring when it is not compressed, is 50mm.",
    "It is a coiled spring wire. The wire has a radius of 2.58mm, with a coil diameter of 82mm, and has a free length of 39mm. There is an 6mm pitch between the coils of the wires.",
    "The spring is a coiled metal with a 40mm coil diameter and an 6mm pitch. Its free length is 53mm, with a 1.61mm wire radius.",
    "The spring is a helical coil with a diameter of 32mm and a pitch of 5mm. Its free length, when not under compression, is 55mm. The wire used in the spring has a radius of 1.62mm.",
    "An illustration of a stapler",
    "A 3d drawing of a circular object",
    "A 3d rendering of a camera",
    "An isometric view of a whistle",
    "An illustration of a piece of concrete with holes",
    "A 3d image of four rings",
    "A 3d image of a circular disc with a central hole and four smaller surrounding holes",
    "A 3d rendering of the letter t",
    "A 3d drawing of a pipe fitting",
    "An illustration of an open wrench with a wrench on it",
    "An illustration of a metal object with a hole.",
    "An illustration of a pillar with a cross on it.",
    "An isometric view of a concrete structure.",
    "An illustration of three stacked rings with same scale",
    "Create a sphere with 100mm^3",
    "Create a sphere with 250mm^3",
    "Create a cylinder with 100mm^3 and 5mm radius",
    "Create a rectangular cuboid with 40mm^2 surface and 10m height.",
    "Three identical rectangular sheets placed vertically, arranged in parallel and evenly spaced, the total volume of the model is 300 mm^3.",
    "Create a 100kg sphere with 10kg/mm^3 density material.",
    "Create a 100kg cylinder with metal.",
    "Create a 100kg cylinder with plastic.",
    "The metal 3D shape is a teardrop-like piece with two circular holes. One large near the broader end and one small near the narrower end.",
    "The plastic 3D shape is a teardrop-like piece with two circular holes. one large near the broader end and one small near the narrower end.",
    "An ice cube and metal ball bigger than the cube.",
    "A cube and metal ball than the cube.",
    "Small box that goes inside a bigger box.",
    "Small box that perfectly fits a big box.",
    "Two pieces of gears that perfectly fit.",
    "Three metal gears that mesh together.",
    "Cylinder pillar that can withstand 10 Mpa.",
    "Cylinder pillar that can withstand 100 Mpa.",
    "Metal Cylinder pillar that can withstand 10 Mpa.",
    "Plastic Cylinder pillar that can withstand 10 Mpa.",
    "Smallest Metal Cylinder pillar that can withstand 10 Mpa.",
    "Smallest Metal Cylinder pillar that can withstand 100 Mpa.",
    "Optimized Metal Cylinder pillar that can withstand 10 Mpa.",
    "Optimized Metal Cylinder pillar that can withstand 100 Mpa.",
    "Airfoil that can be used for propller blade",
    "Airfoil that can be used for plane wing",
    "An optimized Airfoil that can be used for plane wing"
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
        print(f"\n🚀 Processing Prompt {idx}/{len(PROMPTS_LIST) - 1}...")
        
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