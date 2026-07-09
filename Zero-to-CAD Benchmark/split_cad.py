import os
import cadquery as cq
import re

def render_code(code, output_path, filename):
    """Executes the CAD code and saves it as an STL. If it fails, saves an error note."""
    # Skip if code block is empty
    if not code.strip():
        print(f"  -> ⚠️ Skipping {filename}: Code block is empty.")
        error_path = output_path.replace('.stl', '_error.txt')
        with open(error_path, 'w') as f:
            f.write("Error: Code block is empty.\n")
        return

    try:
        local_vars = {}
        # Execute the CadQuery script
        exec(code.strip(), globals(), local_vars)
        
        # Check for the 'result' variable and export
        if 'result' in local_vars:
            cq.exporters.export(local_vars['result'], output_path)
            print(f"  -> ✅ Saved: {filename}")
        else:
            print(f"  -> ❌ Failed {filename}: The model didn't define a 'result' variable. Saving error note.")
            error_path = output_path.replace('.stl', '_error.txt')
            with open(error_path, 'w') as f:
                f.write("Error: The model didn't define a 'result' variable.\n\nCode attempted:\n" + code)
            
    except Exception as e:
        print(f"  -> ❌ Error rendering {filename}: {e}. Saving error note.")
        error_path = output_path.replace('.stl', '_error.txt')
        with open(error_path, 'w') as f:
            f.write(f"Error details:\n{e}\n\nCode attempted:\n" + code)

def process_combined_file(input_file):
    # 1. Define and create your two folders
    dir_direct = "stls_without_reasoning"
    dir_reasoning = "stls_with_reasoning"
    
    os.makedirs(dir_direct, exist_ok=True)
    os.makedirs(dir_reasoning, exist_ok=True)

    # 2. Read the huge text file
    if not os.path.exists(input_file):
        print(f"⚠️ Could not find '{input_file}'. Please check the filename.")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 3. Split the file into chunks by "PROMPT #X:"
    # This regex splits the text and captures the prompt number
    prompt_chunks = re.split(r'={20,}\s*PROMPT #(\d+):', content)

    # prompt_chunks[0] is everything before the first prompt (usually empty)
    # prompt_chunks[1] is the prompt number (e.g., "1")
    # prompt_chunks[2] is the text and code for prompt 1, and so on...
    
    # If no prompts were found, exit
    if len(prompt_chunks) < 3:
        print("⚠️ No PROMPT markers found. Check the formatting.")
        return

    total_prompts = len(prompt_chunks) // 2
    print(f"\n--- Found {total_prompts} Prompts in the file. Starting Rendering... ---\n")

    # 4. Loop through the chunks
    for i in range(1, len(prompt_chunks), 2):
        prompt_num = int(prompt_chunks[i])
        block_text = prompt_chunks[i+1]
        
        # Format the filename with leading zeros (e.g., prompt_001.stl)
        filename = f"prompt_{prompt_num:03d}.stl"
        print(f"\n[Processing Prompt #{prompt_num}]")

        # 5. Use Regex to extract the code under SHEET 1
        # It looks for SHEET 1, grabs everything after it, until it hits SHEET 2
        sheet1_match = re.search(r'--- SHEET 1: DIRECT OUTPUT ---\s+(.*?)(?=--- SHEET 2: REASONING ASKED OUTPUT ---)', block_text, re.DOTALL)
        
        # Use Regex to extract the code under SHEET 2
        # It looks for SHEET 2, grabs everything after it, until it hits the next === line or end of file
        # FIXED: Added the '=' before '{20,}' so it properly matches 20+ equals signs
        sheet2_match = re.search(r'--- SHEET 2: REASONING ASKED OUTPUT ---\s+(.*?)(?=={20,}|\Z)', block_text, re.DOTALL)

        # 6. Render SHEET 1 (Without Reasoning)
        if sheet1_match:
            print("  [Sheet 1 - Direct Output]")
            out_path = os.path.join(dir_direct, filename)
            render_code(sheet1_match.group(1), out_path, filename)
        else:
            print("  -> ⚠️ No 'SHEET 1' found for this prompt.")

        # 7. Render SHEET 2 (With Reasoning)
        if sheet2_match:
            print("  [Sheet 2 - Reasoning Asked Output]")
            out_path = os.path.join(dir_reasoning, filename)
            render_code(sheet2_match.group(1), out_path, filename)
        else:
            print("  -> ⚠️ No 'SHEET 2' found for this prompt.")

    print(f"\n🎉 Finished! All STLs have been sorted into '{dir_direct}' and '{dir_reasoning}'.")

if __name__ == "__main__":
    # UPDATE THIS to the exact name of your massive text file
    INPUT_FILE = "benchmark_results.txt" 
    
    process_combined_file(INPUT_FILE)