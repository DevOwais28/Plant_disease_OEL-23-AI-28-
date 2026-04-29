from random import sample
import pandas as pd
import cv2
import tensorflow as tf
import os
import matplotlib.pyplot as plt

dataset_path = r"D:\Plant_disease(23-AI_28)\PlantVillage"
dataset_folders = os.listdir(dataset_path)
print(dataset_folders)

# for folder in dataset_folders:
#     folder_path = os.path.join(dataset_path, folder)
#     if os.path.isdir(folder_path):
#         for file in os.listdir(folder_path):
#             if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
#                 img_path = os.path.join(folder_path, file)
#                 img = cv2.imread(img_path)
#                 if img is not None:
#                     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#                     resized_img = tf.image.resize(img, (224, 224))
#                     cv2.imwrite(img_path, gray_img)
#                     # print(f"Converted to grayscale: {img_path}")


# Get first image for testing
sample_folder = os.path.join(dataset_path, dataset_folders[0])
sample_file = [f for f in os.listdir(sample_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))][0]
sample_img_path = os.path.join(sample_folder, sample_file)

# Load and preprocess image
img = cv2.imread(sample_img_path)
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply edge detection
laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
plt.imshow(laplacian)
plt.show()

train_ds = tf.keras.utils.image_dataset_from_directory(
    "D:\Plant_disease(23-AI_28)\PlantVillage",
    image_size=(224, 224),
    batch_size=32,
    validation_split=0.2,
    subset="training",
    seed=123,
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "D:\Plant_disease(23-AI_28)\PlantVillage",
    image_size=(224, 224),
    batch_size=32,
    validation_split=0.2,
    subset="validation",
    seed=123,
    shuffle=True
)

model_flat = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(224, 224, 3)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])

model_flat.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model_cnn = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])

model_cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model_flat.fit(train_ds, validation_data=val_ds, epochs=5)

model_cnn.fit(train_ds, validation_data=val_ds, epochs=5)

print(model_flat.history.accuracy)
print(model_cnn.history.val_accuracy)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    weights='imagenet',
    include_top=False
)

# Unfreeze the entire base model
base_model.trainable = True

# Freeze all layers EXCEPT the last 20
for layer in base_model.layers[:-20]:
    layer.trainable = False


inputs = tf.keras.Input(shape=(224, 224, 3))

x = base_model(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
# Dropout for regularization
x = layers.Dropout(0.2)(x)
# Output layer — 4 classes
outputs = layers.Dense(4, activation='softmax')(x)


lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',   # track validation loss
    factor=0.5,           # reduce LR by half
    patience=3,           # wait 3 epochs before reducing
    min_lr=1e-6,          # lowest LR limit
    verbose=1
)

model_frozen = models.Model(inputs, outputs, name='MobileNetV2_Frozen')
model_frozen.summary()

model_frozen.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history_frozen = model_frozen.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=[lr_scheduler]
)