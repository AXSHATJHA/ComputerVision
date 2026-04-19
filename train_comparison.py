import os
import json
import itertools
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, models, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import kagglehub
import warnings
warnings.filterwarnings("ignore")

print("Downloading dataset...")
path = kagglehub.dataset_download("iamsouravbanerjee/animal-image-dataset-90-different-animals")
path = os.path.join(path, "animals", "animals")

data = {"imgpath": [], "labels": []}
category = os.listdir(path)
for folder in category:
    folderpath = os.path.join(path, folder)
    if not os.path.isdir(folderpath): continue
    filelist = os.listdir(folderpath)
    for file in filelist:
        fpath = os.path.join(folderpath, file)
        data["imgpath"].append(fpath)
        data["labels"].append(folder)

df = pd.DataFrame(data)
lb = LabelEncoder()
df['encoded_labels'] = lb.fit_transform(df['labels'])

train_df, Temp_df = train_test_split(df, train_size=0.70, shuffle=True, random_state=124)
valid_df, test_df = train_test_split(Temp_df, train_size=0.70, shuffle=True, random_state=124)
train_df = train_df.reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

BATCH_SIZE = 15
IMAGE_SIZE = (224, 224)
num_classes = len(set(train_df['labels']))

def get_generators(preprocess_func):
    generator = ImageDataGenerator(preprocessing_function=preprocess_func)
    
    train_images = generator.flow_from_dataframe(
        dataframe=train_df, x_col='imgpath', y_col='labels',
        target_size=IMAGE_SIZE, color_mode='rgb', class_mode='categorical',
        batch_size=BATCH_SIZE, shuffle=True, seed=42
    )
    
    val_images = generator.flow_from_dataframe(
        dataframe=valid_df, x_col='imgpath', y_col='labels',
        target_size=IMAGE_SIZE, color_mode='rgb', class_mode='categorical',
        batch_size=BATCH_SIZE, shuffle=False
    )
    
    test_images = generator.flow_from_dataframe(
        dataframe=test_df, x_col='imgpath', y_col='labels',
        target_size=IMAGE_SIZE, color_mode='rgb', class_mode='categorical',
        batch_size=BATCH_SIZE, shuffle=False
    )
    return train_images, val_images, test_images

augment = tf.keras.Sequential([
  layers.RandomFlip("horizontal"),
  layers.RandomRotation(0.15),
  layers.RandomZoom(0.15),
  layers.RandomContrast(0.15),
], name='AugmentationLayer')

models_to_evaluate = [
    {
        "name": "EfficientNetB3",
        "app_model": tf.keras.applications.EfficientNetB3,
        "preprocess": tf.keras.applications.efficientnet.preprocess_input
    },
    {
        "name": "MobileNetV2",
        "app_model": tf.keras.applications.MobileNetV2,
        "preprocess": tf.keras.applications.mobilenet_v2.preprocess_input
    },
    {
        "name": "ResNet50",
        "app_model": tf.keras.applications.ResNet50,
        "preprocess": tf.keras.applications.resnet50.preprocess_input
    }
]

# We will export all data to this dict
dashboard_data = {}

os.makedirs('dashboard', exist_ok=True)

for m_config in models_to_evaluate:
    m_name = m_config["name"]
    print(f"\n==== Training {m_name} ====")
    
    train_gen, val_gen, test_gen = get_generators(m_config["preprocess"])
    
    pretrained_model = m_config["app_model"](
        input_shape=(224, 224, 3), include_top=False, weights='imagenet', pooling='max'
    )
    for layer in pretrained_model.layers:
        layer.trainable = False
        
    inputs = layers.Input(shape=(224,224,3))
    x = augment(inputs)
    pretrain_out = pretrained_model(x, training=False)
    x = layers.Dense(256)(pretrain_out)
    x = layers.Activation(activation="relu")(x)
    x = BatchNormalization()(x)
    x = layers.Dropout(0.45)(x)
    x = layers.Dense(num_classes)(x)
    outputs = layers.Activation(activation="softmax", dtype=tf.float32)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(0.0005), loss='categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(
        train_gen, steps_per_epoch=len(train_gen),
        validation_data=val_gen, validation_steps=len(val_gen),
        epochs=3, # using 3 epochs for fast execution
        verbose=1
    )
    
    res = model.evaluate(test_gen, verbose=0)
    test_loss, test_acc = res[0], res[1]
    
    y_true = test_gen.classes
    y_pred = np.argmax(model.predict(test_gen), axis=1)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    dashboard_data[m_name] = {
        "history": history.history,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "f1_score": float(macro_f1)
    }
    
    # Dump incrementally
    with open('dashboard/metrics.json', 'w') as f:
        json.dump(dashboard_data, f, indent=4)

print("\nAll models trained and exported successfully!")
