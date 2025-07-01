# src/train.py

import tensorflow as tf
import numpy as np
import os
from . import config
from .data_loader import load_and_preprocess_data
from .model import build_model

def main():
    # 1. Set seed for reproducibility
    np.random.seed(config.RANDOM_STATE)
    tf.random.set_seed(config.RANDOM_STATE)

    # 2. Load and prepare data
    print("Loading data...")
    train_ds, val_ds, _, class_weights = load_and_preprocess_data()

    # 3. Build the model
    print("Building model...")
    model = build_model()
    model.summary()

    # 4. Define callbacks for efficient training
    # ModelCheckpoint: Save the best model based on validation accuracy
    checkpoint_path = os.path.join(config.SAVED_MODELS_DIR, "best_model.h5")
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    )

    # EarlyStopping: Stop training if validation accuracy doesn't improve
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5, # Number of epochs with no improvement
        restore_best_weights=True,
        verbose=1
    )

    # ReduceLROnPlateau: Reduce learning rate when a metric has stopped improving
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2, # Factor by which the learning rate will be reduced
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    # 5. Train the model
    print("Starting training...")
    history = model.fit(
        train_ds,
        epochs=config.EPOCHS,
        validation_data=val_ds,
        class_weight=class_weights,
        callbacks=[model_checkpoint, early_stopping, reduce_lr]
    )

    print("\nTraining complete.")
    print(f"Best model saved to {checkpoint_path}")

if __name__ == '__main__':
    if not os.path.exists(config.SAVED_MODELS_DIR):
        os.makedirs(config.SAVED_MODELS_DIR)
    main()