# Minor-Project-2024
Develop an automated system to detect traffic accidents in real-time using surveillance camera footage.

Technologies Used: Python,YOLO, OpenCV, TensorFlow, Keras, PyTorch, NumPy, Pandas
Developed and fine-tuned convolutional neural networks (CNNs) for object detection and scene classification.
Trained models using large datasets to improve detection accuracy and robustness.
Table of Contents
Project Overview
Technologies Used
Installation
Usage
Model Architecture
Data Collection and Preprocessing
Training the Model
Real-Time Video Processing
Results
Project Overview
Develop an automated system to detect traffic accidents in real-time using surveillance camera footage.

Technologies Used
Python
YOLO
OpenCV
TensorFlow
Keras
PyTorch
NumPy
Pandas
Installation
Clone the repository:
git clone https://github.com/meCeltic/Machine-learning-Projects/tree/master/Accident-Detection-Model
Navigate to the project directory:
cd accident-detection-system
Create a virtual environment and activate it:
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
Install the required packages:
pip install -r requirements.txt
Model Architecture
The CNN model used for accident detection is designed with the following layers:

Convolutional layers
MaxPooling layers
Fully connected layers
Softmax activation for output classification
Data Collection and Preprocessing
Data was collected from surveillance footage of traffic scenes.
Images were annotated with accident and non-accident labels.
Preprocessing steps include resizing, normalization, and splitting into training/validation/test sets.
Training the Model
The model was trained using:

YOLO for object detection.
Real-Time Video Processing
OpenCV was used to capture real-time video feed and overlay accident detection results using the trained YOLO model.

Results
The model achieved an accuracy of 90% on the test dataset. Below are some visualizations and example outputs from the real-time monitoring system.
