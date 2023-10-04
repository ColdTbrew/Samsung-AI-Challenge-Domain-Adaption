import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool

# Specify the folder where the CSV files are located
folder_path = 'source/'
sample = "sample_submission.csv"
csv_save_dir = "csvs"
csv_name = 'esemble01'
img_save_dir = 'esemble01'

# List all CSV files in the folder
file_names = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Threshold for the minimum number of votes required to select a class
threshold = 1  # Adjust this threshold as needed

class_intensity = {
    0: 255,
    1: 40,
    2: 60,
    3: 80,
    4: 100,
    5: 120,
    6: 140,
    7: 160,
    8: 100,
    9: 200,
    10: 120,
    11: 170,
    255: 230
}
def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

# Function to process a single group
def process_group(group_index):
    # Initialize an empty array to store class counts for each pixel
    class_counts = np.zeros((540, 960, len(class_intensity)), dtype=np.uint8)
    for file_name in file_names:
        # Create the full path to the CSV file
        file_path = os.path.join(folder_path, file_name)

        df = pd.read_csv(file_path)
        group_df = df[group_index * num_samples_per_group: (group_index + 1) * num_samples_per_group]

        img_id = group_df.iloc[0]['id']
        # Create an image for combining masks with class-specific intensity levels
        mask_image = np.zeros((540, 960), dtype=np.uint8)

        for index, row in group_df.iterrows():
            mask_rle = row['mask_rle']

            if mask_rle != -1:
                mask = rle_decode(mask_rle, (540, 960))
                # Increment the count for the corresponding class
                class_index = index % 12
                class_counts[:, :, class_index] += mask

    # Determine the final class for each pixel based on the threshold
    final_mask_value = np.argmax(class_counts, axis=2)
    final_mask_value[class_counts.max(axis=2) < threshold] = 255  # Assign -1 to pixels below the threshold
    #마스킹 완성
    mask = np.array(final_mask_value)
    for class_id in range(12):
        class_mask = (mask == class_id).astype(np.uint8)
        if np.sum(class_mask) > 0:  # If mask exists, encode
            mask_rle = rle_encode(class_mask)
            result.append(mask_rle)

        else:  # If mask doesn't exist, append -1
            result.append(-1)
    
    # Save the image
    output_image_path = os.path.join(img_save_dir, f'mask_image_{group_index}.png')
    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir)
    plt.imsave(output_image_path, final_mask_value, cmap='viridis', vmin=0, vmax=len(class_intensity) - 1)

if __name__ == '__main__':
    sample_df = pd.read_csv(sample)
    num_samples = len(sample_df)
    num_samples_per_group = 12
    num_groups = num_samples // num_samples_per_group

    result = []  # List to store mask_rle results

    with Pool(processes=12) as pool:  # You can adjust the number of processes as needed
        for _ in tqdm(pool.imap_unordered(process_group, range(num_groups)), total=num_groups):
            pass

    submit = pd.read_csv(sample)
    submit['mask_rle'] = result
    if not os.path.exists(csv_save_dir):
        os.makedirs(csv_save_dir)
    submit.to_csv(os.path.join(csv_save_dir, csv_name + '.csv'), index=False)
