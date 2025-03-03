import os
import shutil

# Define paths
source_root = "/home/houhao/Downloads/multimodal_dataset"
dest_root = "/home/houhao/Downloads/cityscapes_format"

image_dir = os.path.join(source_root, "polL_color")
gt_dir = os.path.join(source_root, "GT")
list_dir = os.path.join(source_root, "list_folder")

# Define Cityscapes-like paths
leftImg8bit_dir = os.path.join(dest_root, "leftImg8bit")
gtFine_dir = os.path.join(dest_root, "gtFine")

# Ensure output directories exist
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(leftImg8bit_dir, split), exist_ok=True)
    os.makedirs(os.path.join(gtFine_dir, split), exist_ok=True)

# Process each split
for split in ["train", "val", "test"]:
    with open(os.path.join(list_dir, f"{split}.txt"), "r") as f:
        lines = f.read().splitlines()

    for line in lines:
        # Source file paths
        image_file = os.path.join(image_dir, f"{line}.png")
        gt_file = os.path.join(gt_dir, f"{line}.png")

        # Destination directories
        city_name = line.split("_")[0]  # e.g., outscene1208_2 -> outscene1208_2
        os.makedirs(os.path.join(leftImg8bit_dir, split, city_name), exist_ok=True)
        os.makedirs(os.path.join(gtFine_dir, split, city_name), exist_ok=True)

        # Destination file paths
        new_image_name = f"{line}_leftImg8bit.png"
        new_gt_name = f"{line}_gtFine_labelIds.png"

        # Copy and rename files
        shutil.copy(image_file, os.path.join(leftImg8bit_dir, split, city_name, new_image_name))
        shutil.copy(gt_file, os.path.join(gtFine_dir, split, city_name, new_gt_name))

print("Dataset converted to Cityscapes format successfully!")
