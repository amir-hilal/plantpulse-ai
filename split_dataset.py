import os
import shutil
from sklearn.model_selection import train_test_split

# Path to your dataset (where all the class folders are located)
data_dir = './PlantVillage'  # Your dataset is in the root directory

# Path to the new directory where you want to save the split data
split_data_dir = './data'  # New split data will go into the ./data folder

# Define the split ratios
split_ratio = [0.7, 0.2, 0.1]  # 70% train, 20% val, 10% test

# Create train, val, test directories
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(split_data_dir, split), exist_ok=True)

# Iterate through each class folder in the dataset
classes = os.listdir(data_dir)
for class_name in classes:
    class_dir = os.path.join(data_dir, class_name)
    if os.path.isdir(class_dir):
        images = os.listdir(class_dir)

        # Split data into train, validation, and test sets
        train_images, temp_images = train_test_split(images, test_size=1-split_ratio[0], random_state=42)
        val_images, test_images = train_test_split(temp_images, test_size=split_ratio[2]/(split_ratio[1] + split_ratio[2]), random_state=42)

        # Create class directories inside train, val, test folders
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(split_data_dir, split, class_name), exist_ok=True)

        # Move images to the corresponding split folders
        for img in train_images:
            shutil.copy(os.path.join(class_dir, img), os.path.join(split_data_dir, 'train', class_name, img))
        for img in val_images:
            shutil.copy(os.path.join(class_dir, img), os.path.join(split_data_dir, 'val', class_name, img))
        for img in test_images:
            shutil.copy(os.path.join(class_dir, img), os.path.join(split_data_dir, 'test', class_name, img))

print("Dataset has been split into train, val, and test sets.")
