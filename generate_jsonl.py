from utils import get_caption_from_idx, get_new_caption_from_idx, get_path_from_idx
from extract_frames import get_frames_across_video_list
from tqdm import tqdm
import os
import cv2
import json 

NUM_VIDEOS = 10000 # start with 10k videos
NUM_FRAMES = 8
kind = 'rewritten' # 'original' or 'rewritten'

invalid_idxs = [799] # Idxs where reading from the video fails

# For the first round of annotations, we are testing original captions vs old captions
def save_frames(idx, frames):
    os.makedirs(f"datasets/webvid/frames/{idx}", exist_ok=True)
    for i, frame in enumerate(frames):
        output_path = f"datasets/webvid/frames/{idx}/{i}.png"
        if os.path.exists(output_path):
            continue
        # Save RGB image (numpy array right now)
        cv2.imwrite(output_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

for idx in tqdm(range(NUM_VIDEOS)):
    try:
        video_path = get_path_from_idx(idx)
        frames = get_frames_across_video_list([video_path], NUM_FRAMES)
        save_frames(idx, frames)
    except KeyboardInterrupt:
        exit()
    except:
        invalid_idxs.append(idx)
        print("Failed to read video at index:", idx)

# Now we need to generate the jsonl file
# Structure: {"id": "val-0000000300", "source": "ucf101", "conversations": [{"images": ["val/BabyCrawling/v_BabyCrawling_g21_c04.0.jpg", "val/BabyCrawling/v_BabyCrawling_g21_c04.1.jpg", "val/BabyCrawling/v_BabyCrawling_g21_c04.2.jpg", "val/BabyCrawling/v_BabyCrawling_g21_c04.3.jpg", "val/BabyCrawling/v_BabyCrawling_g21_c04.4.jpg", "val/BabyCrawling/v_BabyCrawling_g21_c04.5.jpg", "val/BabyCrawling/v_BabyCrawling_g21_c04.6.jpg", "val/BabyCrawling/v_BabyCrawling_g21_c04.7.jpg"], "user": "Classify the video into one of the following classes: ApplyEyeMakeup, ApplyLipstick, Archery, BabyCrawling, BalanceBeam, BandMarching, BaseballPitch, Basketball, BasketballDunk, BenchPress.", "assistant": "BabyCrawling"}]}
def get_jsonl_entry(idx, split='train', kind='original'):
    instruction = "Describe what is happening in the video."
    source = f"{kind}-webvid10m"
    frames = sorted(os.listdir(f"datasets/webvid/frames/{idx}"))
    frames = [f"{idx}/{f}" for f in frames]
    if kind == 'original':
        caption = get_caption_from_idx(idx)
    elif kind == 'rewritten':
        caption = get_new_caption_from_idx(idx)
    else:
        raise ValueError("Invalid kind:", kind, "is not 'original' or 'rewritten'")

    return {
        "id": f"{split}-{str(idx).zfill(10)}",
        "source": source,
        "conversations": [
            {
                "images": frames,
                "user": instruction,
                "assistant": caption
            }
        ]
    }

# do 80/20 train/val split
split_idx = int(0.9 * NUM_VIDEOS)
path = "datasets/webvid/webvid10m_train.jsonl"
if kind == "rewritten":
    path = "datasets/webvid/rewritten_webvid10m_train.jsonl"
print("Writing to jsonl file", path, "...")
with open(path, "w") as f:
    for idx in list(range(NUM_VIDEOS))[:split_idx]:
        if idx in invalid_idxs:
            continue
        f.write(json.dumps(get_jsonl_entry(idx, split='train', kind=kind)))
        f.write("\n")
path = path.replace("train", "val")
with open(path, "w") as f:
    for idx in list(range(NUM_VIDEOS))[split_idx:]:
        if idx in invalid_idxs:
            continue
        f.write(json.dumps(get_jsonl_entry(idx, split='val', kind=kind)))
        f.write("\n")
