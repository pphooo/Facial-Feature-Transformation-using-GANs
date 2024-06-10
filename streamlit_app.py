import cv2
import streamlit as st
import numpy as np
from PIL import Image

# โหลดโมเดล haarcascade สำหรับ face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(image):
    # แปลงภาพจาก RGB เป็น grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # ตรวจจับใบหน้า
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    return faces

def draw_faces(image, faces):
    # วาดกรอบรอบใบหน้า
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return image

# ตั้งค่า Streamlit app
st.title("Face Detection App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # อ่านภาพจากการอัพโหลด
    image = Image.open(uploaded_file)
    image = np.array(image.convert('RGB'))
    
    # ตรวจจับใบหน้า
    faces = detect_faces(image)
    
    # วาดกรอบรอบใบหน้า
    image_with_faces = draw_faces(image.copy(), faces)
    
    # แสดงผลลัพธ์
    st.image(image_with_faces, caption='Detected Faces', use_column_width=True)
    st.write(f"Found {len(faces)} face(s)")

