import cv2
import streamlit as st
from PIL import Image
import numpy as np
from io import BytesIO

st.header('Face Detection with OpenCV')
image = st.file_uploader('Upload an image', type=['png', 'jpg', 'jpeg'])

face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def convert_image(img):
  buf = BytesIO()
  img.save(buf, format="PNG")
  byte_im = buf.getvalue()
  return byte_im

if image is not None:
  img_pil = Image.open(image)
  st.write('Original Image')
  st.image(img_pil)

  img_np = np.array(img_pil)
  img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
  gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

  faces = face_detector.detectMultiScale(gray, 1.6, 5)

  for (x,y,w,h) in faces:  
    cv2.rectangle(img_bgr, (x,y), (x+w,y+h), (0,255,0), 2)
  img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
  st.image(img_rgb)
  out = Image.fromarray(img_rgb)
  st.download_button('Download output image', convert_image(out), 'output.png', 'image/png')
  

