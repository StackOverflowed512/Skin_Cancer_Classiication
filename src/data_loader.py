# src/data_loader.py

import tensorflow as tf
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from . import config

def load_and_preprocess_data():
    """
    Loads the HAM10000 dataset, preprocesses it, and splits it into
    training, validation, and test sets. It also calculates class weights
    to handle class imbalance.

    Returns:
        - train_ds, val_ds, test_ds: tf.data.Dataset objects for model training.
        - class_weights: A dictionary of weights for each class.
    """
    # 1. Load metadata and create image paths
    df = pd.read_csv(config.METADATA_PATH)
    image_paths = {os.path.splitext(f)[0]: os.path.join(p, f)
                   for p in [config.IMAGE_DIR_PART1, config.IMAGE_DIR_PART2]
                   for f in os.listdir(p)}
    df['image_path'] = df['image_id'].map(image_paths.get)
    df = df.dropna(subset=['image_path']) # Remove entries with no image

    # 2. Encode labels
    df['label'] = df['dx'].map(config.CLASS_TO_INT)

    # 3. Split the data
    X = df['image_path']
    y = df['label']

    # Stratified split to maintain class distribution
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=config.RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=config.RANDOM_STATE, stratify=y_temp
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    # 4. Calculate class weights for handling imbalance
    class_weights_array = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = {i: w for i, w in enumerate(class_weights_array)}
    print("Calculated Class Weights:", class_weights)

    # 5. Create tf.data.Dataset objects
    train_ds = tf.data.Dataset.from_tensor_slices((X_train.values, y_train.values))
    val_ds = tf.data.Dataset.from_tensor_slices((X_val.values, y_val.values))
    test_ds = tf.data.Dataset.from_tensor_slices((X_test.values, y_test.values))

    # 6. Define preprocessing and augmentation functions
    def parse_image(filepath, label):
        image = tf.io.read_file(filepath)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [config.IMG_SIZE, config.IMG_SIZE])
        # ResNet50 specific preprocessing
        image = tf.keras.applications.resnet50.preprocess_input(image)
        label = tf.one_hot(label, depth=config.NUM_CLASSES)
        return image, label

    def augment(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        # Add rotation
        image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
        return image, label

    # 7. Build the data pipelines
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = (
        train_ds
        .map(parse_image, num_parallel_calls=AUTOTUNE)
        .cache()
        .shuffle(buffer_size=1000)
        .map(augment, num_parallel_calls=AUTOTUNE)
        .batch(config.BATCH_SIZE)
        .prefetch(buffer_size=AUTOTUNE)
    )

    val_ds = (
        val_ds
        .map(parse_image, num_parallel_calls=AUTOTUNE)
        .batch(config.BATCH_SIZE)
        .cache()
        .prefetch(buffer_size=AUTOTUNE)
    )

    test_ds = (
        test_ds
        .map(parse_image, num_parallel_calls=AUTOTUNE)
        .batch(config.BATCH_SIZE)
        .cache()
        .prefetch(buffer_size=AUTOTUNE)
    )

    return train_ds, val_ds, test_ds, class_weights