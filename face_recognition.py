from flask import Flask, request, jsonify
import face_recognition
from PIL import Image
import numpy as np
import requests
from io import BytesIO

app = Flask(__name__)

def load_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return np.array(img)

@app.route('/compare_faces', methods=['POST'])
def compare_faces():
    data = request.json
    image_url1 = data['imageUrl1']
    image_url2 = data['imageUrl2']

    img1 = load_image_from_url(image_url1)
    img2 = load_image_from_url(image_url2)

    # Find face encodings for each image
    face_encodings1 = face_recognition.face_encodings(img1)
    face_encodings2 = face_recognition.face_encodings(img2)

    if not face_encodings1 or not face_encodings2:
        return jsonify({"error": "No faces found in one or both images"}), 400

    # Compare the first face encoding from each image
    result = face_recognition.compare_faces([face_encodings1[0]], face_encodings2[0])
    return jsonify({"isIdentical": result[0]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
