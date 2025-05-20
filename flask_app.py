from flask import Flask, request, send_file, jsonify
import cv2
import numpy as np
import tempfile

app = Flask(__name__)

# Haar cascade faylini yuklash
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "Rasm yuborilmadi"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Fayl nomi yo'q"}), 400

    # Faylni o'qish
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Yuzlarni aniqlash
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Yuzlar atrofiga to‘rtburchak chizish
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Vaqtinchalik rasm faylini saqlash
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    cv2.imwrite(temp_file.name, image)

    return send_file(temp_file.name, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
