import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras import layers, Model
from tensorflow.keras.applications.efficientnet import preprocess_input

import kagglehub

print("1. Loading Dataset Infrastructure...")
path = kagglehub.dataset_download("iamsouravbanerjee/animal-image-dataset-90-different-animals")
path = os.path.join(path, "animals", "animals")

data = {"imgpath": [], "labels": []}
category = sorted(os.listdir(path))
for folder in category:
    folderpath = os.path.join(path, folder)
    if not os.path.isdir(folderpath): continue
    for file in os.listdir(folderpath):
        fpath = os.path.join(folderpath, file)
        data["imgpath"].append(fpath)
        data["labels"].append(folder)

df = pd.DataFrame(data)
lb = LabelEncoder()
df['encoded_labels'] = lb.fit_transform(df['labels'])

# Match the notebook's exact random states for identical Test-Set allocation securely!
train_df, Temp_df = train_test_split(df, train_size=0.70, shuffle=True, random_state=124)
valid_df, test_df = train_test_split(Temp_df, train_size=0.70, shuffle=True, random_state=124)
test_df = test_df.reset_index(drop=True)

num_classes = len(lb.classes_)

print("2. Reconstructing Model Architecture...")
pretrained_model = tf.keras.applications.EfficientNetB3(
    input_shape=(224, 224, 3), include_top=False, weights=None, pooling='max'
)
inputs = layers.Input(shape=(224,224,3))

# NOTE: Since we are running inference, we DO NOT apply the Training Augmentation Block
pretrain_out = pretrained_model(inputs, training=False)
x = layers.Dense(256)(pretrain_out)
x = layers.Activation(activation="relu")(x)
x = BatchNormalization()(x)
x = layers.Dropout(0.45)(x)
x = layers.Dense(num_classes)(x)
outputs = layers.Activation(activation="softmax", dtype=tf.float32)(x)

model = Model(inputs=inputs, outputs=outputs)
print("3. Connecting Loaded Weights...")
model.load_weights('./checkpoints/my_checkpoint.weights.h5')

print("4. Fetching Inference Sequences...")
# Shuffle test df to get random variants natively
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

correct_samples = []
incorrect_samples = []

# Loop sequentially to find matches and mismatches
for idx in range(len(test_df)):
    row = test_df.iloc[idx]
    
    img = tf.keras.utils.load_img(row['imgpath'], target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array_batch = tf.expand_dims(img_array, 0)
    
    # Needs efficientnet preprocessing!
    processed_input = preprocess_input(img_array_batch.numpy())
    
    # Predict
    preds = model.predict(processed_input, verbose=0)
    pred_idx = np.argmax(preds[0])
    confidence = preds[0][pred_idx] * 100
    pred_label = lb.inverse_transform([pred_idx])[0]
    true_label = row['labels']
    
    sample = {
        'image': img_array / 255.0, # scale strictly for matplotlib viewing
        'true': true_label,
        'pred': pred_label,
        'conf': confidence,
        'is_correct': (true_label == pred_label)
    }
    
    if sample['is_correct'] and len(correct_samples) < 4:
        correct_samples.append(sample)
    elif not sample['is_correct'] and len(incorrect_samples) < 2:
        incorrect_samples.append(sample)
        
    if len(correct_samples) == 4 and len(incorrect_samples) == 2:
        break

# Combine and visually shuffle
final_samples = correct_samples + incorrect_samples
random.shuffle(final_samples)

print("5. Plotting Display Board...")
plt.figure(figsize=(15, 8))
plt.suptitle("Model Evaluation: Ground Truth vs Inference Outcomes", fontsize=20, fontweight='bold', y=0.95)

for i in range(6):
    sample = final_samples[i]
    plt.subplot(2, 3, i + 1)
    
    plt.imshow(sample['image'])
    plt.axis("off")
    
    color = "green" if sample['is_correct'] else "maroon"
    title_text = f"True: {sample['true']}\nPred: {sample['pred']} ({sample['conf']:.1f}%)"
    
    plt.title(title_text, color=color, fontweight='bold', fontsize=13)

plt.tight_layout()
plt.subplots_adjust(top=0.85)

save_path = "prediction_samples.png"
plt.savefig(save_path, dpi=300)
print(f"Extraction Successful! Matrix mapped into {save_path}")
