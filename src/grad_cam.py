# src/grad_cam.py

import tensorflow as tf
import numpy as np
import cv2

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generates a Grad-CAM heatmap.
    """
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam(image_path, model, last_conv_layer_name='conv5_block3_out'):
    """
    Loads an image, processes it, and overlays the Grad-CAM heatmap.
    """
    # Preprocess the image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(config.IMG_SIZE, config.IMG_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    processed_img = tf.keras.applications.resnet50.preprocess_input(img_array_expanded)

    # Generate heatmap
    heatmap = make_gradcam_heatmap(processed_img, model, last_conv_layer_name)

    # Superimpose heatmap on original image
    heatmap = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * 0.4 + img_array
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    # Display
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(tf.keras.preprocessing.image.array_to_img(img_array))
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(superimposed_img)
    plt.title("Grad-CAM Heatmap")
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    # Example Usage: Update evaluate.py or create a new script to use this
    from . import config
    
    model_path = os.path.join(config.SAVED_MODELS_DIR, "best_model.h5")
    model = tf.keras.models.load_model(model_path)
    
    # Find the last convolutional layer name in ResNet50
    # You can print model.summary() to find the exact name
    last_conv_layer = [layer for layer in model.layers[1].layers if 'conv' in layer.name][-1].name
    
    # Find a sample test image
    df = pd.read_csv(config.METADATA_PATH)
    sample_image_id = df[df['dx'] == 'mel']['image_id'].iloc[0] # Example: a melanoma image
    sample_image_path = os.path.join(config.IMAGE_DIR_PART1, f"{sample_image_id}.jpg") # Assuming it's in part 1
    if not os.path.exists(sample_image_path):
         sample_image_path = os.path.join(config.IMAGE_DIR_PART2, f"{sample_image_id}.jpg")

    display_gradcam(sample_image_path, model, last_conv_layer)