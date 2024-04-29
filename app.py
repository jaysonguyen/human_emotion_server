from flask import Flask, request, jsonify, Response, render_template
import cv2
from flask_cors import CORS
import numpy as np
from keras.models import model_from_json
import base64
import json

app = Flask(__name__)
CORS(app)

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load model từ file JSON
json_file = open('model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# Load trọng số vào mô hình
emotion_model.load_weights("model.h5")
print("Loaded model from disk")
#
@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})

    image = request.files['image'].read()
    npimg = np.frombuffer(image, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    emotions = []

    for (x, y, w, h) in faces:
        roi_gray = gray_frame[y:y+h, x:x+w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)

        emotion_prediction = emotion_model.predict(cropped_img)
        emotion_probabilities = emotion_prediction[0] * 100  # Chuyển đổi sang phần trăm

        # Chuyển đổi từng giá trị từ float32 sang float
        emotion_probabilities = [float(prob) for prob in emotion_probabilities]

        # Tạo danh sách các cảm xúc kèm theo tỉ lệ tương ứng
        emotion_result = {emotion_dict[i]: round(emotion_probabilities[i], 2) for i in range(len(emotion_dict))}
        emotions.append(emotion_result)

    # In nội dung phản hồi trước khi trả về
    response_content = {'emotions': emotions}
    print("Response:", emotion_prediction)

    # Trả về danh sách các cảm xúc với tỉ lệ tương ứng
    return jsonify(response_content)


# @app.route('/detect_emotion_webcam', methods=['POST'])
# def detect_emotion_webcam():
#     # Receive image data from the webcam
#     image_data = request.json['image']
#
#     # Decode base64 image data
#     decoded_image = base64.b64decode(image_data.split(',')[1])
#     nparr = np.frombuffer(decoded_image, np.uint8)
#     frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
#     faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
#
#     emotions = []
#
#     for (x, y, w, h) in faces:
#         roi_gray = gray_frame[y:y+h, x:x+w]
#         cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
#
#         emotion_prediction = emotion_model.predict(cropped_img)
#         emotion_probabilities = emotion_prediction[0] * 100  # Convert to percentage
#
#         # Convert each value from float32 to float
#         emotion_probabilities = [float(prob) for prob in emotion_probabilities]
#
#         # Create a dictionary of emotions with corresponding probabilities
#         emotion_result = {emotion_dict[i]: round(emotion_probabilities[i], 2) for i in range(len(emotion_dict))}
#         emotions.append(emotion_result)
#
#     # Return list of emotions with corresponding probabilities
#     response_content = {'emotions': emotions}
#     return jsonify(response_content)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
