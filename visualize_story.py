import pandas as pd
import requests
import os
import decord
import cv2
import json
import matplotlib.pyplot as plt

webvid_csv = pd.read_csv("datasets/webvid/webvid_val.csv")

def caption_to_url(caption):
    # Find caption in "name" column of webvid_csv
    # TODO: Make soft match for caption
    row = webvid_csv[webvid_csv["name"] == caption]
    if row.empty:
        raise ValueError("Caption not found in dataset")

    return row["contentUrl"].values[0]

def download_video(url, location="videos/"):
    # Download video from URL
    response = requests.get(url)
    video_location = location + url.split("/")[-1]
    if os.path.exists(video_location):
        return video_location
    with open(video_location, 'wb') as f:
        f.write(response.content)
    return video_location

def extract_middle_frame(video_location):
    # Extract middle frame from video
    frame_location = video_location.replace(".mp4", ".jpg").replace(
        "videos/", "frames/"
    )
    if os.path.exists(frame_location):
        return frame_location
    vr = decord.VideoReader(video_location)
    middle_frame_idx = len(vr) // 2
    middle_frame = vr[middle_frame_idx].asnumpy()
    # Convert from RGB to BGR (since OpenCV uses BGR)
    middle_frame_bgr = cv2.cvtColor(middle_frame, cv2.COLOR_RGB2BGR)

    # Save the frame as an image
    cv2.imwrite(frame_location, middle_frame_bgr)
    return frame_location

def get_middle_frame_paths(captions):
    middle_frame_paths = []
    for caption in captions:
        url = caption_to_url(caption)
        video_location = download_video(url)
        middle_frame = extract_middle_frame(video_location)
        middle_frame_paths.append(middle_frame)
    return middle_frame_paths

def get_middle_frames_from_story_path(story_path):
    with open(story_path, 'r') as file:
        story_data = json.load(file)
    captions = [item.get('caption') for item in story_data.get('sequence', [])]
    middle_frame_paths = get_middle_frame_paths(captions)
    return middle_frame_paths

def visualize_story(story_path, save_path="story_visualization.png"):
    middle_frame_paths = get_middle_frames_from_story_path(story_path)
    
    # Place all frames in a grid for visualization (top left to bottom right)
    num_frames = len(middle_frame_paths)
    num_cols = 4
    num_rows = -(-num_frames // num_cols)  # Ceiling division
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        if i < num_frames:
            img = plt.imread(middle_frame_paths[i])
            ax.imshow(img)
            ax.axis("off")
        else:
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    return save_path

if __name__ == "__main__":
    os.makedirs("videos", exist_ok=True)
    os.makedirs("frames", exist_ok=True)
    os.makedirs("story_visualizations-st-guided", exist_ok=True)
    for idx in range(20):
        try:
            story_path = visualize_story(f"stories/st-end_guided_story_{idx}.json", save_path=f"story_visualizations-st-guided/unguided_story_{idx}.png")
        except:
            print("Error with story:", idx)
            continue
