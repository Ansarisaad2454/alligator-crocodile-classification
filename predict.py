import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

# --- SETTINGS ---
data_dir = os.path.join(os.getcwd(), 'dataset')
img_height, img_width = 128, 128
batch_size = 32

# List of your saved models to process
model_files = [
    'cnn_model.h5',
    'efficientnetb0_model.h5', 
    'resnet50_model.h5', 
    'mobilenetv2_model.h5'
]

# --- 1. LOAD VALIDATION DATA ---
print("📂 Loading Validation Data...")
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=False # Important: Keep order so labels match predictions!
)

# Extract Labels and Images
y_true = []
all_images = []
for images, labels in val_ds:
    y_true.extend(labels.numpy())
    all_images.extend(images.numpy())

y_true = np.array(y_true) # True answers (0=Alligator, 1=Crocodile)

# --- 2. FUNCTION TO PLOT MATRIX ---
def plot_matrix(cm, model_name):
    # Calculate counts
    tn, fp, fn, tp = cm.ravel()
    
    # Labels for the boxes (Like your image)
    labels = [f"TN\n{tn}", f"FP\n{fp}", f"FN\n{fn}", f"TP\n{tp}"]
    labels = np.asarray(labels).reshape(2,2)
    
    # Plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', cbar=False, 
                xticklabels=['Alligator', 'Crocodile'], 
                yticklabels=['Alligator', 'Crocodile'],
                annot_kws={"size": 16, "weight": "bold"})
    
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('Actual Label', fontsize=12)
    plt.title(f'Confusion Matrix: {model_name}', fontsize=14)
    
    # Save file
    filename = f"confusion_matrix_{model_name}.png"
    plt.savefig(filename)
    print(f"✅ Saved: {filename}")
    plt.close()

# --- 3. LOOP THROUGH MODELS ---
for model_file in model_files:
    if not os.path.exists(model_file):
        print(f"⚠️ Skipping {model_file} (File not found)")
        continue

    print(f"\n🚀 Processing {model_file}...")
    
    # Load Model
    model = tf.keras.models.load_model(model_file)
    
    # Predict
    print("   Making predictions...")
    y_pred_prob = model.predict(np.array(all_images), verbose=0)
    y_pred = (y_pred_prob > 0.5).astype("int32") # Convert to 0 or 1
    
    # Generate Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot & Save
    plot_matrix(cm, model_file.split('.')[0]) # Removes .h5 from name

print("\n🎉 All Confusion Matrices generated!")