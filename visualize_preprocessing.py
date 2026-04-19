import os
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
import kagglehub

print("Searching for a dataset image...")
path = kagglehub.dataset_download("iamsouravbanerjee/animal-image-dataset-90-different-animals")
path = os.path.join(path, "animals", "animals")
categories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

img_path = None
while not img_path:
    cat = random.choice(categories)
    cat_path = os.path.join(path, cat)
    files = os.listdir(cat_path)
    if files:
        img_path = os.path.join(cat_path, random.choice(files))

# Load Raw Image
raw_img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
raw_array = tf.keras.utils.img_to_array(raw_img)
raw_array_batch = tf.expand_dims(raw_array, 0) # Create Batch Dimension

# We will define a few distinct demonstration layers so the user can easily see what happens!
flip_layer = layers.RandomFlip("horizontal")
rot_layer = layers.RandomRotation(0.35) # Increased for demonstration
zoom_layer = layers.RandomZoom(0.4) # Increased for demonstration
contrast_layer = layers.RandomContrast(0.6) # Increased for demonstration

# Generate distinct states
flip_batch = flip_layer(raw_array_batch, training=True)
rot_batch = rot_layer(raw_array_batch, training=True)
zoom_batch = zoom_layer(raw_array_batch, training=True)
contrast_batch = contrast_layer(raw_array_batch, training=True)

# Define full block
combine_block = tf.keras.Sequential([flip_layer, rot_layer, zoom_layer, contrast_layer])
all_batch = combine_block(raw_array_batch, training=True)

# Plot configuration
plt.figure(figsize=(15, 10))
plt.suptitle(f"Data Augmentation Demonstration - Class: {cat}", fontsize=18, fontweight='bold')

# Plot 1: Raw
plt.subplot(2, 3, 1)
plt.title("Raw Base Image", fontsize=12)
plt.imshow(raw_img)
plt.axis("off")

# Plot 2: Flipped
plt.subplot(2, 3, 2)
plt.title("Horizontal Flip", fontsize=12)
plt.imshow(flip_batch[0].numpy().astype("uint8"))
plt.axis("off")

# Plot 3: Rotated
plt.subplot(2, 3, 3)
plt.title("Random Rotation", fontsize=12)
plt.imshow(rot_batch[0].numpy().astype("uint8"))
plt.axis("off")

# Plot 4: Zoomed
plt.subplot(2, 3, 4)
plt.title("Random Cropping/Zoom", fontsize=12)
plt.imshow(zoom_batch[0].numpy().astype("uint8"))
plt.axis("off")

# Plot 5: Contrast
plt.subplot(2, 3, 5)
plt.title("Random Contrast Variance", fontsize=12)
plt.imshow(contrast_batch[0].numpy().astype("uint8"))
plt.axis("off")

# Plot 6: ALL
plt.subplot(2, 3, 6)
plt.title("All Augmentations Pipeline", fontsize=12)
plt.imshow(all_batch[0].numpy().astype("uint8"))
plt.axis("off")

plt.tight_layout()
plt.savefig("preprocessing_comparison_advanced.png", dpi=300)
print("Successfully saved preprocessing_comparison_advanced.png!")
