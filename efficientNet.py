import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
import matplotlib.pyplot as plt

# dataset folder name
data_dir = 'dataset'

# image size and training settings
img_height, img_width = 224, 224   # EfficientNetB0 needs 224x224 images
batch_size = 16
epochs = 10

# Loading dataset (train and validation)
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# load pretrained EfficientNetB0 model
base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(img_height, img_width, 3)
)

# freeze EfficientNetB0 layers
base_model.trainable = False

# Building EfficientNetB0 model
model = models.Sequential([
    # normalize image pixels
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),

    # pretrained EfficientNetB0 model
    base_model,

    # convert feature maps to single vector
    layers.GlobalAveragePooling2D(),

    # fully connected layer
    layers.Dense(64, activation='relu'),

    # output layer (binary classification)
    layers.Dense(1, activation='sigmoid')
])

# compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# model training
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# save trained model
model.save('efficientnetb0_model.h5')
print("Model saved as 'efficientnetb0_model.h5'")

# plot accuracy and loss graphs
plt.figure(figsize=(10,4))

# accuracy graph
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# loss graph
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
