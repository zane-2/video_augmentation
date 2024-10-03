import openai
from openai import OpenAI
from utils import get_new_caption_path_from_idx, get_caption_from_idx, get_new_caption_from_idx, get_path_from_idx
from tqdm import tqdm
import os
from video_clip import get_clip_video_features, get_clip_text_features
import numpy as np

with open('keys/openai.key', 'r') as f:
    openai.api_key = f.readline().strip()

def single_rewrite(caption):
    prompt = "Rewrite the following stock footage descriptors into complete sentences.  Be sure that the resulting sentences sound natural and remove specifics about cameras.\n\nFor example:\nINPUT: 3d render of inky injections into water with luma matte. blue ink on white background 5\nOUTPUT: Blue ink injections onto a white background.\n\nINPUT: \"Swimming in the pool ,slow motion 120 fps,handheld camera balanced steady shot \" \nOUTPUT: A person swimming in the pool.\n\nINPUT: Aerial drone isle of wight needles england travel sunrise\nOUTPUT: The sun rises over the Isle of Wight Needles in England.\n\nINPUT: CAPTION\nOUTPUT: "
    prompt = prompt.replace("CAPTION", caption)
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
            "role": "system",
            "content": [
                {
                "text": "You are a helpful assistant.",
                "type": "text"
                }
            ]
            },
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": prompt
                }
            ]
            },
        ],
        temperature=0.2,
        max_tokens=876,
        top_p=0.5,
        frequency_penalty=0,
        presence_penalty=0,
        response_format={
            "type": "text"
        }
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    num_to_recaption = 10000
    for idx in tqdm(range(num_to_recaption)):
        caption = get_caption_from_idx(idx)
        new_caption_path = get_new_caption_path_from_idx(idx)

        # check if exists
        if os.path.exists(new_caption_path):
            continue
        
        os.makedirs(os.path.dirname(new_caption_path), exist_ok=True)

        new_caption = single_rewrite(caption)
        with open(new_caption_path, "w") as f:
            f.write(new_caption)

    # Calculate the similarity scores between the original captions and the videos
    
    # get video features
    print("Calculating video features...")
    video_features = []
    invalid_idxs = [] # Where reading from the file fails
    for idx in tqdm(range(num_to_recaption)):
        try:
            video_features.append(get_clip_video_features(get_path_from_idx(idx), save=False).cpu())
        except:
            invalid_idxs.append(idx)
            print("Failed to read video at index:", idx)
    video_features = np.stack(video_features)

    # get original captions
    original_captions = []
    new_captions = []
    for idx in tqdm(range(num_to_recaption)):
        if idx in invalid_idxs:
            continue
        original_captions.append(get_caption_from_idx(idx))
        new_captions.append(get_new_caption_from_idx(idx))
    
    # get caption embeddings
    orig_embeds = []
    new_embeds = []
    print("Calculating caption embeddings...")
    for orig_cap, new_cap in tqdm(zip(original_captions, new_captions)):
        orig_embeds.append(get_clip_text_features(orig_cap).cpu())
        new_embeds.append(get_clip_text_features(new_cap).cpu())
    
    orig_embeds = np.stack(orig_embeds)
    new_embeds = np.stack(new_embeds)

    # calculate similarity scores
    orig_scores = np.dot(video_features, orig_embeds.T)
    new_scores = np.dot(video_features, new_embeds.T)

    # Get the average along the diagonal
    orig_scores = np.diag(orig_scores).mean()
    new_scores = np.diag(new_scores).mean()


    print("Original scores:", orig_scores)
    print("New scores:", new_scores)


    




    