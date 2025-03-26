# 🎯 Computer Vision Master Project – Plant, Flower & Face Recognition Systems

## 📌 Project Overview

This multi-phase computer vision project applies deep learning, image processing, and real-time recognition to various domains like **botanical research**, **entertainment**, and **facial identity recognition**. It spans five major modules across two projects, leveraging CNNs, Transfer Learning, TensorFlow, GUI, and OpenCV.

---

## 🌿 Project 1: Botanical Image Classification (Plants & Flowers)

### 🧪 Domain: Botanical Research

University X is conducting image-based research to automate plant and flower classification using image datasets and CNNs.

### 📦 Datasets:
- **Plant Seedlings Dataset** (Kaggle): 12 classes
- **Oxford 17 Flower Dataset** (via tflearn)
- `Prediction - Flower.jpg`: custom flower image

### 🔍 Objectives:
- Build CNN for seedling classification
- Train multiple models (ML, NN, CNN) for flower classification
- Apply OpenCV filters (grayscale, sharpen, blur)
- Predict unseen flower image
- Compare model performance

### 🛠️ Key Components:
- Data wrangling, preprocessing, resizing, normalization
- Label encoding, data augmentation
- CNN and supervised learning models
- Evaluation using accuracy, precision, recall
- Image prediction using best model

---

## 🧠 Project 2: Face Detection, Metadata Extraction & Face Recognition

### 📺 Domain: Entertainment & Facial Biometrics

Company X aims to integrate AI into movie streaming apps for actor recognition, metadata tagging, and real-time face identification.

---

### 📌 Part A: Face Mask Detection (Semantic Segmentation)

- **Data:** `images.npy` (Images + Masked Face Regions)
- Built U-Net style segmentation using **MobileNet** as the encoder
- Custom **Dice coefficient** and **Dice loss** implemented
- Evaluated on test images and visualized predicted vs actual face masks

---

### 📌 Part B: Face Metadata Extraction (Haar Cascades)

- **Data:** `training_images` folder (unlabeled face images)
- Used **Haarcascade Frontal Face Detector** (OpenCV)
- Detected faces in all images, extracted bounding box metadata
- Saved extracted features (filename, dimensions, coordinates) to `face_metadata.csv`

---

### 📌 Part C: Face Recognition using VGG Face & SVM

- **Data:** `PINS` dataset (10,770 celebrity face images from Pinterest)
- Generated embeddings using **VGG Face (`vgg_face_weights.h5`)**
- Created metadata and feature vectors for all identities
- Applied **PCA** for dimensionality reduction
- Trained **SVM classifier** to identify people
- Tested recognition on `Benedict_Cumberbatch9.jpg` and `Dwayne_Johnson4.jpg`

---

## ⚙️ Tools, Libraries & Skills Used

- **Languages:** Python  
- **Deep Learning:** TensorFlow, Keras, TFLearn  
- **Libraries:** NumPy, Pandas, Matplotlib, Seaborn, OpenCV, Scikit-learn  
- **Concepts:**
  - Computer Vision (Image Classification, Face Detection)
  - CNNs & Transfer Learning (VGG16, MobileNet)
  - Semantic Segmentation (U-Net style)
  - Image Preprocessing & Augmentation
  - Face Embeddings & Metadata Extraction
  - Dimensionality Reduction (PCA)
  - SVM Classification
  - GUI Application Development (Tkinter)
  - Evaluation Metrics: Accuracy, Dice Score, Precision, Recall

---

## 📁 Repository Structure

<pre>
.
├── Computer Vision Project -1/
│   ├── code/
│   │   ├── CV_1_ISHANT_KUNDRA.ipynb
│   │   └── CV_1_ISHANT_KUNDRA.html
│   └── problem statement/
│       └── CV 1 - Problem_Statement.pdf

├── Computer Vision Project -2/
│   ├── code/
│   │   ├── CV_2_MultiModule_FaceRecognition.ipynb
│   │   └── CV_2_MultiModule_FaceRecognition.html
│   └── problem statement/
│       └── CV 2 - Problem_Statement.pdf
├── README.md
└── .gitignore
</pre>

---

## 💡 Key Learnings

- Developed an end-to-end plant & flower image classification pipeline using CNNs  
- Built facial mask segmentation model using MobileNet + U-Net  
- Extracted face metadata using OpenCV Haar Cascades  
- Generated and used face embeddings for high-accuracy recognition  
- Applied PCA and SVM for lightweight, high-performance identity prediction  
- Integrated prediction into usable formats (GUI-ready and CSV output)

---

## ✍️ Author

**Ishant Kundra**  
📧 [ishantkundra9@gmail.com]
🎓 Master’s in Computer Science | AIML Track
