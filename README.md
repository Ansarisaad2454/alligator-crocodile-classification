# Alligator vs. Crocodile Classification

A Deep Learning project to classify images of Alligators and Crocodiles using various Convolutional Neural Network (CNN) architectures and Transfer Learning models.

## 📌 Project Overview
This project compares multiple deep learning models to distinguish between alligators and crocodiles, which are often difficult to differentiate visually. The project includes data preprocessing, model training with multiple architectures, and a web application for real-time predictions.

## 📂 Features
* **Data Cleaning:** Automated script to remove corrupted images.
* **Multiple Architectures:** Implemented and compared:
    * Custom CNN
    * VGG16
    * ResNet50
    * MobileNetV2
    * EfficientNetB0
* **Web Application:** A user-friendly interface (`app.py`) to upload images and get predictions.
* **Performance Analysis:** Visualizations of accuracy and loss (`prediction_results.png`).

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** TensorFlow/Keras, NumPy, Pandas, Matplotlib, PIL
* **Interface:** Flask / Streamlit (inferred from `app.py`)

## 📂 File Structure

```text
├── Dataset/                  # Training images (Alligator/Crocodile)
├── test_dataset/             # Testing images
├── app.py                    # Web application for deployment
├── clean_data.py             # Script to remove bad files from dataset
├── predict.py                # Script to run predictions on new images
├── correct_incorrect.py      # Analysis of correct vs incorrect predictions
│
├── cnn.py                    # Custom CNN Model Training
├── VGG16.py                  # VGG16 Transfer Learning Training
├── efficientNet.py           # EfficientNet Training
├── mobileNetV2.py            # MobileNetV2 Training
├── restNet50.py              # ResNet50 Training
│
├── *.h5                      # Saved trained models (e.g., cnn_model.h5)
├── *.png                     # Confusion matrices and accuracy plots
└── Report.docx               # Full project report



🚀 How to Run
1. Clone the Repository
Bash
git clone [https://github.com/Ansarisaad2454/alligator-crocodile-classification.git](https://github.com/Ansarisaad2454/alligator-crocodile-classification.git)
cd alligator-crocodile-classification
2. Install Dependencies
Make sure you have the required libraries installed:

Bash
pip install tensorflow numpy pandas matplotlib flask
3. Train a Model
To train a specific model (e.g., MobileNetV2), run:

Bash
python mobileNetV2.py
4. Run the Web App
To start the interface and test with your own images:

Bash
python app.py
📊 Results
The project includes accuracy plots and confusion matrices comparing the different models.

Best Model: [Mention which model worked best, e.g., MobileNetV2]

Accuracy: [Mention accuracy, e.g., 92%]

See prediction_results.png and cnn.png for visual performance metrics.

📝 Author
Saad Ansari GitHub Profile
