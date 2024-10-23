import os
import json
from collections import Counter
import matplotlib.pyplot as plt
from glob import glob
import numpy as np

# Directory containing the story JSON files
story_dir = "stories/st-end*"

# Initialize a counter for the captions
caption_counter = Counter()

# Loop through each file in the directory
for filename in glob(story_dir):
    if filename.endswith(".json"):  # Check if the file is a JSON file
        print("Trying file:", filename)

        # Open and load the JSON file
        with open(filename, 'r') as file:
            story_data = json.load(file)
            
            # Extract captions from the sequence
            for item in story_data.get('sequence', []):
                caption = item.get('caption')
                if caption:  # Ensure the caption exists
                    caption_counter[caption] += 1

def calc_gini(counter):
    # Extract frequencies from the Counter object
    frequencies = np.array(list(counter.values()))
    
    # If all frequencies are zero, the Gini coefficient is zero (no inequality)
    if np.sum(frequencies) == 0:
        return 0.0
    
    # Sort the frequencies in ascending order
    frequencies = np.sort(frequencies)
    
    # Calculate the cumulative sum of frequencies
    cumulative_frequencies = np.cumsum(frequencies)
    
    # Calculate the Gini coefficient
    n = len(frequencies)
    gini = (2 * np.sum((np.arange(1, n + 1) * frequencies))) / (n * np.sum(frequencies)) - (n + 1) / n
    
    return gini

def calc_average(counter):
    frequencies = np.array(list(counter.values()))
    return np.mean(frequencies)

# Extract captions and frequencies
captions, frequencies = zip(*caption_counter.most_common())
gini = round(calc_gini(caption_counter), 4)
average = round(calc_average(caption_counter), 4)
print("GINI Coefficient:", gini)
print("Average Frequency:", average)
# Plotting the frequencies of captions
plt.figure(figsize=(12, 6))
plt.bar(captions, frequencies)
plt.xticks(rotation=90, fontsize=8)
plt.xlabel('Captions')
plt.ylabel('Frequency')
# Make y axis go up by 1 
plt.yticks(range(max(frequencies) + 1))
plt.title('Frequency of Captions in Stories (guided)')
plt.tight_layout()
plt.show()
plt.savefig('st-end_guided_caption_frequencies_20.png')
