import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

# Define directories
train_data_dir = 'dataset/training'
validation_data_dir = 'dataset/testing'

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Print to verify data loading
print(f"Found {train_generator.samples} training images belonging to {train_generator.num_classes} classes.")
print(f"Found {validation_generator.samples} validation images belonging to {validation_generator.num_classes} classes.")

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten

# Load the VGG16 model without the top layers
vgg16_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers of the VGG16 model
for layer in vgg16_base.layers:
    layer.trainable = False

# Define the model
model = Sequential([
    vgg16_base,
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Print the model summary to verify the architecture
#
model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Checkpoint to save the best model
checkpoint = ModelCheckpoint(
    filepath='model/brain_tumor_detector_final.keras',
    save_best_only=True,
    monitor='val_loss',
    mode='min'
)

# Train model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=20,
    callbacks=[checkpoint]
)

# Save the model
model.save('brain_tumor_detector_final.keras')

# Evaluate model performance
loss, accuracy = model.evaluate(validation_generator)
print(f"Validation Accuracy: {accuracy*100:.2f}%")
print(f"Validation Loss: {loss:.4f}")




def process_image(filepath):
    img = image.load_img(filepath, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    return x


img = process_image("C:/Users/saira/Documents/BRAIN TUMOR DETECTION/BRAIN TUMOR DETECTION/dataset/Testing/class1/Te-noTr_0000.jpg")
print(model.predict(img))


print(f"Prediction: {prediction[0][0]}")
if prediction[0][0] > 0.5:
    print("Predicted Class: 1 (Tumor)")
else:
    print("Predicted Class: 0 (No Tumor)")
