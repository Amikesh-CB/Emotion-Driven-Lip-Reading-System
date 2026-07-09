# Emotion-Driven Lip Reading System
An end-to-end Deep Learning project that combines **Facial Emotion Recognition** and **Visual Lip Reading** to improve speech understanding in noisy conversational environments.



## 📌 Overview

Traditional Automatic Speech Recognition (ASR) systems rely heavily on audio input, making them less effective in noisy environments. This project introduces a multimodal framework that integrates:

- Facial Emotion Recognition
- Visual Lip Reading
- CNN-based Feature Extraction
- Multimodal Feature Fusion

By combining facial expressions with lip movements, the system improves contextual understanding of spoken language.



## 🚀 Features

- 😊 Facial Emotion Recognition using CNN
- 👄 Visual Lip Reading
- 📷 Real-time Face Detection using Haar Cascade
- 📊 Confusion Matrix Visualization
- 📈 Training Accuracy & Loss Curves
- 📉 Precision, Recall and F1-Score Evaluation
- 🤖 Multimodal CNN Framework



## 🛠 Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Google Colab



## 📂 Dataset

### Emotion Recognition
- FER2013 Dataset
- 7 Emotion Classes
  - Angry
  - Disgust
  - Fear
  - Happy
  - Neutral
  - Sad
  - Surprise

### Lip Reading
- Lip Reading Dataset (GRID/Kaggle)



## 🧠 Model Architecture

### Emotion Recognition

Sequential CNN consisting of:

- Rescaling Layer
- Convolution Layers
- Max Pooling Layers
- Flatten Layer
- Dense Layer
- Softmax Output Layer

### Lip Reading

CNN-based visual feature extraction followed by classification of lip movement patterns.



## 📌 Workflow

```text
Input Image / Video
        │
        ▼
Face Detection (Haar Cascade)
        │
        ▼
Emotion Recognition CNN
        │
        ▼
Lip Region Extraction
        │
        ▼
Lip Reading CNN
        │
        ▼
Feature Fusion
        │
        ▼
Final Emotion-Aware Speech Prediction
```



## 📈 Results

### Emotion Recognition

- Training Epochs: **10**
- Final Validation Accuracy: **53.98%**

Performance Metrics

| Metric | Value |
|---------|-------|
| Accuracy | 53.98% |
| Precision | 0.5378 |
| Recall | 0.5398 |
| F1 Score | 0.5327 |

The model successfully recognizes facial emotions while showing moderate confusion between visually similar emotions such as Neutral, Fear and Sad.



### Lip Reading

The lip-reading model predicts visual speech classes using lip movement features.

Evaluation includes:

- 20-Class Confusion Matrix
- Normalized Confusion Matrix
- Visual Speech Classification

The proposed framework demonstrates improved contextual understanding by combining emotion recognition with lip-reading features.




# ⚙️ Methodology

The proposed framework follows a multimodal deep learning pipeline that integrates facial emotion recognition and visual lip reading to improve speech understanding in noisy conversational environments.

## Step 1: Data Collection

Two publicly available datasets are used for training and evaluation.

| Dataset | Purpose |
|----------|----------|
| FER2013 | Facial Emotion Recognition |
| Lip Reading Dataset (GRID/Kaggle) | Visual Speech Recognition |

The FER2013 dataset contains grayscale facial images representing seven different emotions, while the lip-reading dataset consists of video frames capturing various spoken words and lip movement patterns.



## Step 2: Data Preprocessing

Before training, all input images undergo several preprocessing operations:

- Convert images to grayscale
- Resize images to a fixed resolution
- Normalize pixel values to the range **[0,1]**
- Apply histogram equalization for contrast enhancement
- Reduce noise and improve feature visibility

These preprocessing techniques improve the robustness of feature extraction and accelerate model convergence.



## Step 3: Face Detection

The system detects human faces using the **OpenCV Haar Cascade Classifier**.

The detected face region is cropped and forwarded to the emotion recognition network.

**Output**

- Face Bounding Box
- Cropped Face Image



## Step 4: Emotion Recognition

A Sequential Convolutional Neural Network (CNN) is trained on the FER2013 dataset to classify facial expressions.

### CNN Architecture

```
Input Image
      │
      ▼
Rescaling Layer
      │
      ▼
Conv2D + ReLU
      │
      ▼
MaxPooling
      │
      ▼
Conv2D + ReLU
      │
      ▼
MaxPooling
      │
      ▼
Flatten
      │
      ▼
Dense Layer
      │
      ▼
Softmax Output
```

The model predicts one of the following emotions:

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise



## Step 5: Lip Region Extraction

After detecting the face, the mouth region is extracted using facial landmarks.

The extracted lip region is resized and normalized before being passed to the lip-reading model.

**Output**

- Cropped Lip Image



## Step 6: Visual Lip Reading

The lip-reading CNN extracts spatial features from lip movements to recognize spoken words without relying on audio signals.

The model learns visual speech patterns (visemes) and predicts the corresponding speech class.



## Step 7: Feature Fusion

The outputs from both CNN models are combined to improve contextual understanding.

```
Emotion Features
        │
        ├────────────┐
        │            │
        ▼            ▼
      Fusion Layer (Concatenation)
               │
               ▼
     Fully Connected Layer
               │
               ▼
 Final Emotion-Aware Speech Prediction
```

Feature fusion enables the system to jointly analyze facial expressions and lip movements, improving recognition performance in challenging environments.



## Step 8: Model Training

The emotion recognition model was trained using:

| Parameter | Value |
|------------|-------|
| Model | Sequential CNN |
| Optimizer | Adam |
| Loss Function | Categorical Crossentropy |
| Epochs | 10 |
| Activation | ReLU & Softmax |

Training progress was monitored using accuracy and loss curves.



## Step 9: Performance Evaluation

The trained models are evaluated using standard classification metrics.

| Metric | Description |
|---------|-------------|
| Accuracy | Overall prediction performance |
| Precision | Correct positive predictions |
| Recall | Ability to detect actual positives |
| F1-Score | Harmonic mean of Precision and Recall |
| Confusion Matrix | Class-wise prediction analysis |
| Normalized Confusion Matrix | Percentage-wise prediction performance |


## 🔄 Complete Pipeline

```
Input Image / Video
        │
        ▼
Image Preprocessing
        │
        ▼
Face Detection
        │
        ▼
Emotion Recognition CNN
        │
        ▼
Lip Region Extraction
        │
        ▼
Lip Reading CNN
        │
        ▼
Feature Fusion
        │
        ▼
Performance Evaluation
        │
        ▼
Final Emotion-Aware Speech Prediction
```



## 📊 Generated Outputs

The project generates:

- Sequential CNN Architecture
- Emotion Confusion Matrix
- Normalized Emotion Confusion Matrix
- Model Accuracy Curve
- Model Loss Curve
- Lip Reading Confusion Matrix
- Normalized Lip Reading Confusion Matrix
- Classification Report
- Precision, Recall & F1 Score



## 📁 Project Structure

```
Emotion-Aware-Visual-Speech-Recognition/
│
├── dataset/
│   ├── FER2013/
│   └── LipReading/
│
├── notebooks/
│   └── Emotion_Aware.ipynb
│
├── models/
│   ├── emotion_model.h5
│   └── lipreading_model.h5
│
├── outputs/
│   ├── confusion_matrix.png
│   ├── normalized_confusion.png
│   ├── accuracy.png
│   ├── loss.png
│   └── classification_report.png
│
├── README.md
└── requirements.txt
```



## ▶️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Emotion-Aware-Visual-Speech-Recognition.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the notebook

```bash
jupyter notebook Emotion_Aware.ipynb
```

or

Open the notebook in **Google Colab**.



## 📚 Applications

- Human-Computer Interaction
- Assistive Communication Systems
- Silent Speech Recognition
- Smart Surveillance
- AI-Based Conversation Systems
- Emotion-Aware Virtual Assistants



## 🔮 Future Improvements

- Transformer-based Lip Reading
- Real-time Video Inference
- Audio + Visual + Emotion Fusion
- Larger Emotion Datasets
- Attention Mechanisms
- Improved Generalization Accuracy



## 📖 References

- Goodfellow et al. — FER2013 Dataset
- LipNet: End-to-End Sentence-level Lipreading
- CREMA-D Dataset
- OpenCV Haar Cascade
- TensorFlow & Keras Documentation



## 👨‍💻 Author

**Amikesh Kesharwani**

B.Tech (Information Technology)

Harcourt Butler Technical University (HBTU), Kanpur

---

⭐ If you found this project useful, consider giving it a star on GitHub!
