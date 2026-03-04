📌 Project Overview

The AI-Based Exam Cheating Detection System is a real-time intelligent monitoring solution designed to detect suspicious student behavior during examinations using Computer Vision and Machine Learning techniques.

This system uses:

Facial landmark detection

Head pose estimation

Face count monitoring

Machine learning classification (Random Forest)

The goal is to automatically identify suspicious movements such as:

Looking away repeatedly

Unusual head movements

Multiple faces appearing in frame

Behavior anomalies

This system is designed to be scalable for:

Online exams

Remote proctoring

National testing boards

Universities and schools

🧠 System Architecture

The system works in 3 major stages:

1️⃣ Face & Landmark Detection (Computer Vision Layer)

We use the pretrained facial landmark model from
dlib

This model detects:

68 facial landmark points

Nose position

Chin position

Eye positions

Face boundaries

This enables head pose and movement tracking.

2️⃣ Feature Engineering (Behavior Analysis Layer)

From the detected landmarks, we extract behavioral features:

nose_chin_angle → detects head tilt

face_count → detects multiple persons

eye_left_x → left eye horizontal movement

eye_right_x → right eye horizontal movement

These features are converted into numerical format for training.

3️⃣ Machine Learning Classification (Decision Layer)

We train a:

🌲 Random Forest Classifier

Why Random Forest?

Handles nonlinear behavior patterns

Works well on small datasets

Reduces overfitting

High interpretability

Good accuracy for behavioral classification

Output:

0 → Normal Behavior

1 → Suspicious Behavior

The trained model is saved as:

models/cheating_model.pkl
📂 Complete Project Structure
ai-exam-cheating-detection/
│
├── dataset/
│   └── head_movement_dataset.csv
│
├── models/
│   └── cheating_model.pkl
│
├── utils/
│   ├── head_pose_estimation.py
│   └── feature_extraction.py
│
├── train_model.py
├── main.py
├── requirements.txt
└── README.md
📁 Folder & File Explanation
📁 dataset/

Contains:

head_movement_dataset.csv

Custom-generated dataset (<25MB) containing:

Feature	Description
nose_chin_angle	Head tilt angle
face_count	Number of faces detected
eye_left_x	Left eye position
eye_right_x	Right eye position
label	0 = Normal, 1 = Suspicious

This dataset is used to train the ML model.

📁 models/

Contains:

cheating_model.pkl

Saved trained Random Forest model using joblib.

This model is loaded during live webcam monitoring.

📁 utils/
head_pose_estimation.py

Calculates head angle using:

Nose landmark

Chin landmark

Used to detect head tilt behavior.

feature_extraction.py

Extracts numerical features from:

Facial landmarks

Face count

These features are passed to ML classifier.

📄 train_model.py

Responsible for:

Loading CSV dataset

Splitting train/test data

Training Random Forest

Evaluating model

Saving trained model

Evaluation Metrics Used:

Accuracy

Precision

Recall

F1 Score

📄 main.py

This is the LIVE SYSTEM.

Functions:

Opens webcam

Detects face

Extracts features

Sends features to trained model

Displays:

"Normal"

"ALERT: Suspicious Activity"

Runs in real-time.

📄 requirements.txt

Contains required Python libraries:

opencv-python
dlib
numpy
pandas
scikit-learn
joblib
imutils
🚀 How to Run the Project
Step 1: Install dependencies
pip install -r requirements.txt
Step 2: Train the Model
python train_model.py

This will create:

models/cheating_model.pkl
Step 3: Run Live Detection
python main.py

Press ESC to exit.

🎯 Key Features

✔ Real-time webcam monitoring
✔ Head movement tracking
✔ Multiple face detection
✔ ML-based suspicious classification
✔ Lightweight dataset (<25MB)
✔ Works offline
✔ Expo-ready live demo

📊 Dataset Information

Custom generated behavioral dataset

5000+ labeled samples

Balanced classes (Normal vs Suspicious)

Size under 25MB

Designed for demonstration and research prototype
