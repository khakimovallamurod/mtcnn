from mtcnn import MTCNN
from mtcnn.utils.images import load_image 
import cv2

detector = MTCNN(device="CPU:0")

def detect_faces(image):
    return detector.detect_faces(image)

def draw_faces(image, result_list):
    for box in result_list:
        x, y, width, height = box['box']
        x2, y2 = x + width, y + height
        cv2.rectangle(image, (x, y), (x2, y2), (0, 255, 0), 3)
    return image

def show_image(image_path):
    image = load_image(image_path)
    result = detect_faces(image)
    result_image = draw_faces(image, result)
    result_image_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
    
    return result_image_bgr

