import cv2
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# Initialize image data generator with rescaling and normalization
train_data_gen = ImageDataGenerator(rescale=1./255, samplewise_center=True, samplewise_std_normalization=True)
validation_data_gen = ImageDataGenerator(rescale=1./255, samplewise_center=True, samplewise_std_normalization=True)

# Preprocess all train images with additional log transformation
train_data_gen_with_log = ImageDataGenerator(rescale=1./255, samplewise_center=True, samplewise_std_normalization=True, preprocessing_function=np.log1p)
train_generator = train_data_gen_with_log.flow_from_directory(
    'data/train',
    target_size=(48, 48),
    batch_size=32,
    color_mode="grayscale",
    class_mode='categorical')

# Preprocess all test images with additional log transformation
validation_data_gen_with_log = ImageDataGenerator(rescale=1./255, samplewise_center=True, samplewise_std_normalization=True, preprocessing_function=np.log1p)
validation_generator = validation_data_gen_with_log.flow_from_directory(
    'data/test',
    target_size=(48, 48),
    batch_size=32,
    color_mode="grayscale",
    class_mode='categorical')

# Create model structure with L2 regularization
emotion_model = Sequential()

emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(0.001), input_shape=(48, 48, 1)))
emotion_model.add(BatchNormalization())
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(0.001)))
emotion_model.add(BatchNormalization())
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.5))

emotion_model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(0.001)))
emotion_model.add(BatchNormalization())
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(0.001)))
emotion_model.add(BatchNormalization())
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.5))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu', kernel_regularizer=l2(0.001)))
emotion_model.add(BatchNormalization())
emotion_model.add(Dropout(0.75))
emotion_model.add(Dense(7, activation='softmax'))

cv2.ocl.setUseOpenCL(False)

# Updated optimizer with new learning_rate argument
emotion_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

# Define early stopping and learning rate reduction callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-7)

# Train the neural network/model with early stopping and learning rate reduction
emotion_model_info = emotion_model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=50,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=[early_stopping, reduce_lr])

# Calculate predictions
y_true = validation_generator.classes
y_pred = emotion_model.predict(validation_generator).argmax(axis=1)

# Calculate evaluation metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1)

# Save metrics to file
score_dict = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1
}

with open("Model_Data/0_score.txt", "w") as score_file:
    for key, value in score_dict.items():
        score_file.write(f"{key}: {value}\n")

# Plot training and validation loss
plt.plot(emotion_model_info.history['loss'])
plt.plot(emotion_model_info.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()
plt.savefig("Model_Data/Graphs/0_loss.png")

# Plot training and validation accuracy
plt.plot(emotion_model_info.history['accuracy'])
plt.plot(emotion_model_info.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.show()
plt.savefig("Model_Data/Graphs/0_accuracy.png")

# Save model structure in a JSON file
model_json = emotion_model.to_json()
with open("emotion_model_0.json", "w") as json_file:
    json_file.write(model_json)

# Save trained model weights in an H5 file
emotion_model.save_weights('emotion_model_0.h5')
