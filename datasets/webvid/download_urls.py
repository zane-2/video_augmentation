from glob import glob
import os
import requests
from tqdm import tqdm
import pandas as pd


partion_csvs = sorted(glob("partitions/*.csv"))

os.makedirs('videos', exist_ok=True)


for csv in tqdm(partion_csvs):
    # Make dir to store the files
    os.makedirs(f'videos/{csv.split("/")[-1].split(".")[0]}', exist_ok=True)
   
    df = pd.read_csv(csv)
    for idx, row in df.iterrows():
        output_path = f'videos/{csv.split("/")[-1].split(".")[0]}/{str(idx).zfill(4)}.mp4'
        url = row["contentUrl"]
        r = requests.get(url, allow_redirects=True)
        open(output_path, 'wb').write(r.content)
    
        
            