from glob import glob
import pandas as pd
import os

webvid_partitions = sorted(glob("datasets/webvid/partitions/*.csv"))

all_urls = []
all_captions = []
partition_lengths = []

for part in webvid_partitions:
    df = pd.read_csv(part)
    all_urls.extend(df["contentUrl"].tolist())
    all_captions.extend(df["name"].tolist())
    partition_lengths.append(len(df))



def get_caption_from_idx(idx):
    if idx >= len(all_captions):
        raise ValueError("Index out of bounds, max index: ", len(all_captions), " requested index: ", idx)
    return all_captions[idx]

def get_url_from_idx(idx):
    if idx >= len(all_urls):
        raise ValueError("Index out of bounds, max index: ", len(all_urls), " requested index: ", idx)
    return all_urls[idx]

def get_path_from_idx(idx):
    if idx >= len(all_urls):
        raise ValueError("Index out of bounds, max index: ", len(all_urls), " requested index: ", idx)
    # see which partition the idx belongs to 
    partition_idx = 0
    while idx >= partition_lengths[partition_idx]:
        idx -= partition_lengths[partition_idx]
        partition_idx += 1
    
    return f'datasets/webvid/videos/{webvid_partitions[partition_idx].split("/")[-1].split(".")[0]}/{str(idx).zfill(4)}.mp4'

def get_new_caption_path_from_idx(idx):
    if idx >= len(all_urls):
        raise ValueError("Index out of bounds, max index: ", len(all_urls), " requested index: ", idx)
    # see which partition the idx belongs to 
    partition_idx = 0
    while idx >= partition_lengths[partition_idx]:
        idx -= partition_lengths[partition_idx]
        partition_idx += 1
    
    return f'datasets/webvid/new_captions/{webvid_partitions[partition_idx].split("/")[-1].split(".")[0]}/{str(idx).zfill(4)}.txt'

def get_new_caption_from_idx(idx):
    if idx >= len(all_urls):
        raise ValueError("Index out of bounds, max index: ", len(all_urls), " requested index: ", idx)
    if not os.path.exists(get_new_caption_path_from_idx(idx)):
        raise ValueError("New caption not found for index: ", idx)
    with open(get_new_caption_path_from_idx(idx), "r") as f:
        s = f.read()
    return s
    
if __name__ == "__main__":
    test_idx = 146
    print(get_caption_from_idx(test_idx))
    print(get_url_from_idx(test_idx))
    print(get_path_from_idx(test_idx))
    print(get_new_caption_path_from_idx(test_idx))
    print(get_new_caption_from_idx(test_idx))
    