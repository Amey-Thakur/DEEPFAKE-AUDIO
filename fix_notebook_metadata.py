import json
import os

notebooks = [
    "DEEPFAKE-AUDIO.ipynb",
    "Source Code/DEEPFAKE-AUDIO.ipynb"
]

for nb_path in notebooks:
    if not os.path.exists(nb_path):
        print(f"Skipping {nb_path} (not found)")
        continue
        
    print(f"Processing {nb_path}...")
    try:
        with open(nb_path, "r", encoding="utf-8-sig") as f:
            nb = json.load(f)
            
        # Fix the 'Invalid Notebook' error: 
        # missing 'state' key in 'metadata.widgets'
        if "metadata" in nb:
            if "widgets" in nb["metadata"]:
                print(f"Removing 'widgets' from metadata in {nb_path}")
                del nb["metadata"]["widgets"]
            else:
                print(f"No 'widgets' key in metadata for {nb_path}")
        else:
            print(f"No 'metadata' key in {nb_path}")

        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=4)
        print(f"Successfully fixed {nb_path}")
    except Exception as e:
        print(f"Error processing {nb_path}: {e}")
