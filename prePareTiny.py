import os
import shutil

# Path to the root of the Tiny ImageNet dataset
data_path = 'tiny-imagenet-200'

# Path to the val directory
val_dir = os.path.join(data_path, 'val')

# Ensure the 'images' directory exists within the 'val' directory
images_dir = os.path.join(val_dir, 'images')
if not os.path.exists(images_dir):
    os.makedirs(images_dir)

# Read the validation annotations to get the class labels for each image
val_annotations_path = os.path.join(val_dir, 'val_annotations.txt')
val_annotations = {}
with open(val_annotations_path, 'r') as f:
    for line in f:
        parts = line.split('\t')
        img_name = parts[0]
        class_name = parts[1]
        val_annotations[img_name] = class_name

# Move validation images to new structure
for img_name, class_name in val_annotations.items():
    class_dir = os.path.join(images_dir, class_name)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)
    img_src_path = os.path.join(images_dir, img_name)
    img_dst_path = os.path.join(class_dir, img_name)
    if os.path.exists(img_src_path):  # Ensure the image exists before moving
        shutil.move(img_src_path, img_dst_path)

# Clean up any leftover images in the 'images' directory
for leftover_file in os.listdir(images_dir):
    file_path = os.path.join(images_dir, leftover_file)
    if os.path.isfile(file_path):
        os.remove(file_path)

print("Dataset restructuring complete.")
