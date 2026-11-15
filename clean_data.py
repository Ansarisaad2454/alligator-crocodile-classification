import os
from PIL import Image

data_dir = 'dataset'

print("Checking dataset for corrupt images...")

bad_images = 0

for folder in os.listdir(data_dir):
    folder_path = os.path.join(data_dir, folder)

    if not os.path.isdir(folder_path):
        continue

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)

        try:
            img = Image.open(file_path)
            img.verify()
        except (IOError, SyntaxError, OSError) as e:
            print(f"Removing corrupt image: {file_path}")
            os.remove(file_path)
            bad_images += 1

print(f"\nDone! Total corrupt images removed: {bad_images}")