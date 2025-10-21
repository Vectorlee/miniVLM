import os
import subprocess
from huggingface_hub import snapshot_download

os.environ["HF_TOKEN"] = "YOUR_HF_TOKEN"

snapshot_download(repo_id="BAAI/SVIT", repo_type="dataset", local_dir="./", allow_patterns="*.zip")

# will get the following data
# data: complex_reasoning.zip conversation.zip  detail_description.zip  referring_qa.zip  svit.zip
# raw: images.zip, images2.zip

# prepare the data:
#  - unzip images.zip images2.zip
#  - move all images into image_data/ directory
#  - unzip conversion.zip, get conversion.json
#  - move conversation.json in the top directory 