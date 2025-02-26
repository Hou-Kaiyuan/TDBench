
import pandas as pd
import os
from vlmeval.smp import encode_image_file_to_base64

# Create LMUData directory if it doesn't exist
lmu_data_dir = os.path.expanduser('~/LMUData')
os.makedirs(lmu_data_dir, exist_ok=True)

# Prepare the data
image_base64 = encode_image_file_to_base64('assets/apple.jpg')
data = pd.DataFrame({
    'index': [0],
    'image': [image_base64],  # Base64 encoded image
    'question': ['What is the object in the image:'],
    'A': ['Apple'],
    'B': ['Banana'],
    'C': ['Pineapple'],
    'D': ['Melon'],
    'answer': ['A']
})

# Save as TSV
tsv_path = os.path.join(lmu_data_dir, 'bevbench.tsv')
data.to_csv(tsv_path, sep='\t', index=False)
print(f"Dataset saved to {tsv_path}")
