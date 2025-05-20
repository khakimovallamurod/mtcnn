from flask import Flask, request, jsonify, send_file
from mtcnn import MTCNN
from mtcnn.utils.images import load_image
import cv2
import numpy as np
import tempfile
import os

app = Flask(__name__)
detector = MTCNN(device="CPU:0")

def detect_faces(image):
    return detector.detect_faces(image)

def draw_faces(image, result_list):
    for box in result_list:
        x, y, width, height = box['box']
        x2, y2 = x + width, y + height
        cv2.rectangle(image, (x, y), (x2, y2), (0, 0, 255), 2)
    return image

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "Rasm fayli kerak. 'image' nomli fayl yuboring."}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Rasm fayli tanlanmagan."}), 400

    # Rasmni numpy arrayga o‘tkazish
    file_bytes = np.frombuffer(file.read(), np.uint8)
    bgr_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # BGR -> RGB, chunki MTCNN RGB formatda ishlaydi
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    # Yuzlarni aniqlash
    results = detect_faces(rgb_image)

    # Chizilgan rasm
    result_image = draw_faces(rgb_image, results)

    # RGB -> BGR, saqlash uchun
    result_image_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)

    # Natijaviy rasmni vaqtinchalik faylga yozish
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    cv2.imwrite(temp_file.name, result_image_bgr)

    return send_file(temp_file.name, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
