# src/model.py

import tensorflow as tf
from . import config

def build_model(fine_tune=False):
    """
    Builds a CNN model using ResNet50 as a pre-trained base.

    Args:
        fine_tune (bool): If True, a portion of the base model layers
                          will be unfrozen for fine-tuning.

    Returns:
        A compiled Keras model.
    """
    # 1. Load the pre-trained base model (ResNet50)
    base_model = tf.keras.applications.ResNet50(
        input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3),
        include_top=False,  # Exclude the final classification layer
        weights='imagenet'
    )

    # 2. Freeze the base model layers
    base_model.trainable = False

    # 3. Add custom classification layers on top
    inputs = tf.keras.Input(shape=(config.IMG_SIZE, config.IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(config.NUM_CLASSES, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    # 4. (Optional) Fine-tuning
    if fine_tune:
        base_model.trainable = True
        # Freeze all layers except the last few convolutional blocks
        fine_tune_at = 143 # Unfreeze from conv5_block1_1_conv onwards
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

    # 5. Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    if fine_tune:
        # Use a lower learning rate for fine-tuning
        optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE / 10)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy',
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )

    return model