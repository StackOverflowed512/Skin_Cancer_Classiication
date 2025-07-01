# src/evaluate.py

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import os
from . import config
from .data_loader import load_and_preprocess_data

def evaluate_model():
    # 1. Load the test dataset
    # We only need the test_ds part of the returned tuple
    _, _, test_ds, _ = load_and_preprocess_data()

    # 2. Load the trained model
    model_path = os.path.join(config.SAVED_MODELS_DIR, "best_model.h5")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please train the model first.")
        return

    print("Loading trained model...")
    model = tf.keras.models.load_model(model_path)

    # 3. Make predictions on the test set
    print("Evaluating on the test set...")
    y_pred_probs = model.predict(test_ds)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # 4. Get true labels
    y_true = np.concatenate([y for x, y in test_ds], axis=0)
    y_true = np.argmax(y_true, axis=1) # Convert from one-hot to integer labels

    # 5. Print Classification Report
    class_names = list(config.CLASSES.keys())
    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # 6. Plot Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    # 7. Calculate and Plot ROC-AUC Score
    # Note: This is for multi-class, using One-vs-Rest strategy
    roc_auc = roc_auc_score(y_true, y_pred_probs, multi_class='ovr', average='macro')
    print(f"\nMacro-average ROC-AUC Score: {roc_auc:.4f}")

    # Plot ROC curve for each class
    plt.figure(figsize=(12, 8))
    for i in range(config.NUM_CLASSES):
        fpr, tpr, _ = roc_curve(y_true, y_pred_probs[:, i], pos_label=i)
        plt.plot(fpr, tpr, label=f'{config.INT_TO_CLASS[i]} (area = {roc_auc_score(y_true == i, y_pred_probs[:, i]):.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == '__main__':
    evaluate_model()