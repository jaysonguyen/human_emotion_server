import numpy as np
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load JSON and create model
json_file = open('emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# Load weights into the new model
emotion_model.load_weights("emotion_model.h5")
print("Loaded model from disk")

# Initialize image data generator with rescaling
test_data_gen = ImageDataGenerator(rescale=1./255)

# Preprocess all test images
test_generator = test_data_gen.flow_from_directory(
        'archive_custom/test',
        target_size=(48, 48),
        batch_size=32,
        color_mode="grayscale",
        class_mode='categorical',
        shuffle=False)  # Ensure data is not shuffled to maintain order for evaluation

# Do prediction on test data
predictions = emotion_model.predict(test_generator)

# Calculate confusion matrix
c_matrix = confusion_matrix(test_generator.classes, predictions.argmax(axis=1))
print("Confusion Matrix:")
print(c_matrix)

# Calculate classification report
report = classification_report(test_generator.classes, predictions.argmax(axis=1), target_names=emotion_dict.values())
print("Classification Report:")
print(report)
