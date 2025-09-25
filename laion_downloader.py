from datasets import load_dataset
import requests
import os
from PIL import Image
from io import BytesIO
import hashlib
import time

# Load the dataset (metadata only)
dataset = load_dataset("laion/relaion2B-en-research-safe", split="train")

# Directory to save images
save_dir = "relaion2B_images"
os.makedirs(save_dir, exist_ok=True)

# Function to download one image safely
def download_image(url, idx, caption):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content)).convert("RGB")
            # Use hash to avoid collisions
            fname = hashlib.sha1(url.encode()).hexdigest() + ".jpg"
            fpath = os.path.join(save_dir, fname)
            img.save(fpath, "JPEG", quality=90)
            return {"file_name": fname, "caption": caption}
    except Exception as e:
        return None
    return None

# Loop through a few samples (⚠️ full dataset = billions, needs cluster/distributed)
metadata = []
for i, sample in enumerate(dataset):
    url = sample["URL"]
    caption = sample["TEXT"]

    result = download_image(url, i, caption)
    if result:
        metadata.append(result)

    # Example: stop after 1000 for demo
    if i > 1000:
        break
    if i % 100 == 0:
        print(f"Processed {i} images...")

# Save metadata
import json
with open("relaion2B_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Done! Images + metadata saved.")
