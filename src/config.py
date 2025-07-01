# src/config.py

import os

# --- Project Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
SAVED_MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')

# --- Dataset Details ---
METADATA_PATH = os.path.join(DATA_DIR, 'HAM10000_metadata.csv')
IMAGE_DIR_PART1 = os.path.join(DATA_DIR, 'ham10000_images_part_1')
IMAGE_DIR_PART2 = os.path.join(DATA_DIR, 'ham10000_images_part_2')

# --- Model & Training Parameters ---
IMG_SIZE = 224  # Input size for ResNet50
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
RANDOM_STATE = 42

# --- Class Mapping ---
# This dictionary maps the abbreviated class names to full, readable names.
# It also helps in encoding labels to integers.
CLASSES = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}

# Create a mapping from class abbreviation to an integer index
CLASS_TO_INT = {label: i for i, label in enumerate(CLASSES.keys())}
INT_TO_CLASS = {i: label for label, i in CLASS_TO_INT.items()}
NUM_CLASSES = len(CLASSES)