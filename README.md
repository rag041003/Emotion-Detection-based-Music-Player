# Emotion-Based Music Player

**First, you need to train the `EmotionDetector.py` Python model and then run the `app.py` file.**

## Project Description

This project is an emotion-based music player designed to enhance the listening experience by selecting music that aligns with the user's emotional state. It uses the Facial Expression Recognition (FER) database and a custom Convolutional Neural Network (CNN) model to detect emotions from user images. Built with Python, TensorFlow/Keras, and OpenCV, this project leverages state-of-the-art machine learning and image processing techniques to deliver a personalized music experience.

## Features

- **Emotion Detection**: Accurately detects emotions from user images using the FER database.
- **Custom CNN Model**: Implements a custom CNN model to achieve precise emotion recognition.
- **Music Recommendation**: Selects and plays music that corresponds to the detected emotion.

## Technologies Used

- Python
- TensorFlow/Keras
- OpenCV
- FER Database

## How It Works

1. **Emotion Detection**: The system captures a user's image and processes it to detect their emotional state.
2. **Model Prediction**: The custom CNN model analyzes the image and predicts the emotion.
3. **Music Selection**: Based on the predicted emotion, the system selects and plays music that corresponds to the user's mood.

## Prerequisites

- Python 3.6 or higher
- TensorFlow
- Keras
- OpenCV
- Numpy

