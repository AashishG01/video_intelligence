import os
import requests
import zipfile
from pathlib import Path

def install_antelopev2():
    # 1. Define the InsightFace model directory
    model_root = Path.home() / ".insightface" / "models"
    model_dir = model_root / "antelopev2"
    zip_path = model_root / "antelopev2.zip"
    
    # Create directory if it doesn't exist
    model_root.mkdir(parents=True, exist_ok=True)

    # 2. Download URL (Mirror link if main fails)
    url = "https://huggingface.co/MonsterMMORPG/tools/resolve/main/antelopev2.zip"

    if not model_dir.exists():
        print(f"⏳ Downloading AntelopeV2 to {zip_path}...")
        response = requests.get(url, stream=True)
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        print("📦 Extracting files...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(model_root)
            
        # Cleanup
        os.remove(zip_path)
        print("✅ AntelopeV2 installed successfully!")
    else:
        print("✅ AntelopeV2 already exists in ~/.insightface/models/")

if __name__ == "__main__":
    install_antelopev2()