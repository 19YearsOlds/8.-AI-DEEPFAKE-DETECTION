from flask import Flask, request, jsonify
import cv2
import numpy as np
from model import detect_deepfake
from dotenv import load_dotenv
import os
load_dotenv()

app = Flask(__name__)

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['file']
    image = cv2.imdecode(np.frombuffer(file.read(), np.unit8), cv2.IMREAD_COLOR)
    result = detect_deepfake(image)
    return jsonify({"deepfake": result})

if __name__ == '__main__':
    app.run(debug=True)