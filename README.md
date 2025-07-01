# Skin Cancer Classification with Deep Learning

This project implements a deep learning pipeline for multi-class classification of skin lesions using dermoscopy images from the HAM10000 dataset. The goal is to assist dermatologists in early and accurate diagnosis of skin cancer by leveraging transfer learning and modern data augmentation techniques.

---

## 📋 Table of Contents

-   [Project Overview](#project-overview)
-   [Dataset](#dataset)
-   [Project Structure](#project-structure)
-   [Setup & Installation](#setup--installation)
-   [Usage](#usage)
-   [Results](#results)
-   [Key Features](#key-features)
-   [Future Work](#future-work)
-   [References](#references)

---

## 📝 Project Overview

-   **Objective:** Build and evaluate a convolutional neural network (CNN) for classifying dermoscopic images into 7 skin lesion categories.
-   **Approach:** Uses transfer learning with ResNet50 as a fixed feature extractor and a custom classifier head.
-   **Frameworks:** TensorFlow/Keras, scikit-learn, pandas, matplotlib, seaborn.

---

## 📊 Dataset

-   **Source:** [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
-   **Classes:** 7 types of skin lesions (e.g., Melanocytic nevi, Melanoma, etc.)
-   **Images:** Over 10,000 dermoscopic images with metadata.

---

## 🗂️ Project Structure

```
ML_Assign/
│
├── data/                  # Dataset (not included in repo, see instructions)
│   ├── HAM10000_metadata.csv
│   ├── ham10000_images_part_1/
│   └── ham10000_images_part_2/
│
├── notebooks/
│   ├── 1_EDA.ipynb        # Exploratory Data Analysis
│   └── train.ipynb        # Model training and evaluation
│
├── src/
│   └── config.py          # Project configuration (paths, constants)
│
├── saved_models/          # Saved model checkpoints
├── reports/
│   └── report_template.md # Final report template
├── requirements.txt       # Python dependencies
└── README.md              # Project overview (this file)
```

---

## ⚙️ Setup & Installation

1. **Clone the repository:**

    ```sh
    git clone https://github.com/StackOverflowed512/Skin_Cancer_Classiication
    cd ML_Assign
    ```

2. **Install dependencies:**

    ```sh
    pip install -r requirements.txt
    ```

3. **Download the HAM10000 dataset:**

    - Place `HAM10000_metadata.csv` and image folders inside the `data/` directory as shown above.

4. **(Optional) Set up a virtual environment:**
    ```sh
    python -m venv venv
    venv\Scripts\activate
    ```

---

## 🚀 Usage

-   **Exploratory Data Analysis:**  
    Open and run `notebooks/1_EDA.ipynb` to explore the dataset, visualize class distributions, and sample images.

-   **Model Training & Evaluation:**  
    Open and run `notebooks/train.ipynb` to preprocess data, train the model, and visualize results.

---

## 🏆 Results

-   **Best Validation Accuracy:** ~66.7%
-   **Techniques Used:**
    -   Class weighting to address class imbalance
    -   Data augmentation (flips, brightness, contrast)
    -   Early stopping, learning rate reduction, and model checkpointing
-   **Evaluation:**
    -   Classification report and confusion matrix
    -   ROC-AUC and Grad-CAM visualizations

See [`reports/report_template.md`](reports/report_template.md) for a detailed summary and visualizations.

---

## ✨ Key Features

-   **Transfer Learning:** ResNet50 backbone with frozen weights.
-   **Custom Classifier Head:** Dense layers with dropout for regularization.
-   **Robust Data Pipeline:** Efficient loading, augmentation, and caching using `tf.data`.
-   **Reproducibility:** Fixed random seeds and clear configuration.

---

## 🔭 Future Work

-   Fine-tune top layers of ResNet50 for improved accuracy.
-   Experiment with advanced augmentations (CutMix, MixUp).
-   Try other architectures (EfficientNet, InceptionV3).
-   Hyperparameter tuning (learning rate, dropout, etc.).
-   Ensemble methods for better robustness.

---

## 📚 References

-   [HAM10000 Dataset on Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
-   [TensorFlow Documentation](https://www.tensorflow.org/)
-   [Keras Applications](https://keras.io/api/applications/)
