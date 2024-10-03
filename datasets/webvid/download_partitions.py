import requests
from tqdm import tqdm
import os

num_paritions = 200

os.makedirs("partitions", exist_ok=True)

for idx in tqdm(range(num_paritions)):
    idx_str = str(idx).zfill(4)

    url = f"https://huggingface.co/datasets/TempoFunk/webvid-10M/raw/main/data/train/partitions/{idx_str}.csv"

    # download the file
    r = requests.get(url, allow_redirects=True)
    open(f'partitions/{idx_str}.csv', 'wb').write(r.content)


