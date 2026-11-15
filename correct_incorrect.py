import tensorflow as tf
import numpy as np
import os
from PIL import Image
import sys

if not os.path.exists('efficientnetb0_model.h5'):
    print('Model file not found. Train the model first.')
    sys.exit(1)

# load best model
model = tf.keras.models.load_model("efficientnetb0_model.h5")

# paths
test_dir = "test_dataset"
correct_dir = "correct_predictions"
incorrect_dir = "incorrect_predictions"

# create folders
os.makedirs(correct_dir, exist_ok=True)
os.makedirs(incorrect_dir, exist_ok=True)

class_names = ["Alligator", "Crocodile"]
img_size = (224, 224)

for label, class_name in enumerate(class_names):
    class_path = os.path.join(test_dir, class_name)

    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)

        image = Image.open(img_path).convert("RGB")
        img = image.resize(img_size)

        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0][0]
        predicted_label = 1 if prediction > 0.5 else 0

        if predicted_label == label:
            image.save(os.path.join(correct_dir, img_name))
        else:
            image.save(os.path.join(incorrect_dir, img_name))

print("Correct and Incorrect images saved!")