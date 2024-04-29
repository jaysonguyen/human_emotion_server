import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
emotion_model = load_model('model/emotion_modelV2.h5')

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Define the emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

while True:
    ret, frame = cap.read()  # Capture frame-by-frame
    if not ret:
        break

    # Convert the captured frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize the frame to 48x48 pixels to match the input size of the model
    resized_frame = cv2.resize(gray, (48, 48))

    # Normalize the pixel values to be in the range [0, 1]
    normalized_frame = resized_frame / 255.0

    # Reshape the frame to match the input shape of the model
    input_frame = np.reshape(normalized_frame, (1, 48, 48, 1))

    # Perform emotion prediction
    predictions = emotion_model.predict(input_frame)

    # Get the index of the predicted emotion
    predicted_emotion_index = np.argmax(predictions)

    # Get the label of the predicted emotion
    predicted_emotion_label = emotion_labels[predicted_emotion_index]

    # Display the predicted emotion on the frame
    cv2.putText(frame, predicted_emotion_label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Emotion Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
