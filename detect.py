import os
import random
import pygame
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
import base64
from flask import send_from_directory

app = Flask(__name__)

# Load model architecture
with open('classifier.json', 'r') as json_file:
    classifier_json = json_file.read()

classifier = model_from_json(classifier_json)

# Load model weights
classifier.load_weights("./classifier_weights.weights.h5")  # Update with the correct model file path

face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)   # 0 for video capture and 1 for single Image capture

class_labels = ['Angry', 'Happy', 'Neutral', 'Sad']
detected_emotion = None

# Function to detect emotion
def detect_emotion(frame):
    global detected_emotion

    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Draw green rectangle around the face
        img_gray = gray[y:y+h, x:x+w]
        img_gray = cv2.resize(img_gray, (64, 64), interpolation=cv2.INTER_AREA)

        if np.sum([img_gray]) != 0:
            img = img_gray.astype('float') / 255.0
            img_arr = img_to_array(img)
            img_arr = np.expand_dims(img_arr, axis=0)
            
            # Make a prediction
            predicted = classifier.predict(img_arr)[0]
            detected_emotion = class_labels[np.argmax(predicted)]
            print("Detected emotion:", detected_emotion)

            label_position = (x, y)
            cv2.putText(frame, detected_emotion, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    cv2.imshow('Face Emotion Detector', frame)

    # Close the camera after processing the image
    cap.release()
    cv2.destroyAllWindows()

# Route to render the index.html template
@app.route('/')
def index():
    return render_template('index1.html')

# Route to capture image for emotion detection
@app.route('/capture', methods=['POST'])
def capture_image():
    data = request.get_json()
    image_data = data['image_data']
    encoded_data = image_data.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    cv2.imwrite('captured_image.jpg', frame)
    detect_emotion(frame)

    # Generate the YouTube search link
    search_query = detected_emotion.lower() + " heal and relief songs"
    youtube_link = f"https://www.youtube.com/results?search_query={search_query}"
    return jsonify({'emotion': detected_emotion, 'youtube_link': youtube_link})

@app.route('/music_player/<emotion>', methods=['GET'])
def music_player(emotion):
    # Define the directory path based on the detected emotion
    directory_path = f"music_list/song/{emotion}"

    # Get the list of songs in the directory
    songs = os.listdir(directory_path)

    # Render the music player HTML page with the list of songs
    return render_template('music_player.html', songs=songs, emotion=emotion)

from flask import send_from_directory

@app.route('/music_list/song/<emotion>/<path:filename>')
def serve_song(emotion, filename):
    return send_from_directory(f'music_list/song/{emotion}', filename)



# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
