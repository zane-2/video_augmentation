import openai
from openai import OpenAI
from tqdm import tqdm
import os
import random
import json

with open('keys/openai.key', 'r') as f:
    openai.api_key = f.readline().strip()

captions = None
with open("prompts/webvid_5k.txt", "r") as f:
    # Read entire txt file into a string
    captions = f.read()

def generate_story(start_caption="", end_caption=""):
    start_caption = "" # Male engineer explaining while holding blueprint
    input_text = "You are an expert videographer.  Given the following captions of unrelated video clips, put together a plausible sequence of events from the captions.  When the video clips are played one after the other (without any captions shown), the viewer can infer that each clip is a continuation of the previous one.  Please quote the exact captions used and remember that each clip contains different scenes with different actors.\n\nFormat your response in a json format, where it is a list of dictionaries, with each dictionary containing a \"caption\" key and a \"reasoning\" key, corresponding to the ground truth captions and your reason for putting them in that specific order.  Include approximately 10 video clips in your output. An example response with only 1 video is shown below. \n[\"sequence\": {\"caption\": \"<CAPTION-1>\", \"reasoning\": \"Caption 1 provides a good starting point for our story about XYZ\"}, ... ]\n\n"
    if start_caption != "":
        input_text = input_text + "Start your story with the following caption: " + start_caption
    if end_caption != "":
        input_text = input_text + "\nEnd your story with the following caption: " + end_caption
    response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
        "role": "system",
        "content": [
            {
            "text": input_text,
            "type": "text"
            }
        ]
        },
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": captions
            }
        ]
        }
    ],
    temperature=1,
    max_tokens=1024,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    response_format={
        "type": "json_object"
    }
    )

    try:
        json_resp = eval(response.choices[0].message.content)
    except Exception as e:
        print(e)
        print(response.choices[0].message.content)
        json_resp = None

    return json_resp

def get_random_caption():
    global captions
    # Get a random caption from the list of captions
    all_captions = captions.split("\\n")
    return random.choice(all_captions)

if __name__ == "__main__":
    guided_style = "start_end"
    if guided_style == "unguided":
        for idx in tqdm(range(20)):
            story = generate_story()
            os.makedirs("stories", exist_ok=True)
            if story is not None:
                # Save with json instead
                with open(f"stories/unguided_story_{idx}.json", "w") as f:
                    json.dump(story, f)
    elif guided_style == "start":
        for idx in tqdm(range(20)):
            start_caption = get_random_caption()
            story = generate_story(start_caption)
            os.makedirs("stories", exist_ok=True)
            if story is not None:
                with open(f"stories/start_guided_story_{idx}.json", "w") as f:
                    json.dump(story, f)
    elif guided_style == "end":
        for idx in tqdm(range(20)):
            end_caption = get_random_caption()
            story = generate_story(start_caption="", end_caption=end_caption)
            os.makedirs("stories", exist_ok=True)
            if story is not None:
                with open(f"stories/end_guided_story_{idx}.json", "w") as f:
                    json.dump(story, f)
    elif guided_style == "start_end":
        for idx in tqdm(range(20)):
            end_caption = get_random_caption()
            start_caption = get_random_caption()
            story = generate_story(start_caption=start_caption, end_caption=end_caption)
            os.makedirs("stories", exist_ok=True)
            if story is not None:
                with open(f"stories/st-end_guided_story_{idx}.json", "w") as f:
                    json.dump(story, f)
