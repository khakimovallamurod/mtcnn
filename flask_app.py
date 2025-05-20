from flask import Flask, request, send_file, jsonify
import cv2
import numpy as np
import tempfile

app = Flask(__name__)

# Haar kaskad faylini yuklab oling (OpenCV bilan birga keladi)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_faces(image):
    # OpenCV faqat BGR formatda ishlaydi
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return faces

def draw_faces(image, faces):
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "Iltimos, 'image' nomli fayl yuboring."}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Rasm tanlanmagan."}), 400

    # Rasmni o‘qish
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    faces = detect_faces(image)
    result_image = draw_faces(image, faces)

    # Natijaviy rasmni vaqtincha faylga yozamiz
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    cv2.imwrite(temp_file.name, result_image)

    return send_file(temp_file.name, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
