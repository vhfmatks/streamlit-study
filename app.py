import streamlit as st
from PIL import Image
import numpy as np
import cv2
from deepface import DeepFace

uploaded_file = st.file_uploader("사진을 업로드하세요", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    st.image(uploaded_file, caption='업로드한 사진', use_container_width=True)
   
    image = Image.open(uploaded_file)
    image_array = np.array(image)

    with st.spinner('얼굴을 감지하는 중입니다...'):
        faces = DeepFace.analyze(img_path=image_array,
                                actions=['age'],
                                detector_backend='retinaface')
    for face in faces:
        x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
        cv2.rectangle(image_array, (x, y), (x+w, y+h), (0, 255, 0), 2)
    st.image(image_array, caption='얼굴 감지 결과', use_container_width=True)
    st.write("감지된 얼굴 수:", len(faces))
    st.write("업로드한 사진의 크기는", uploaded_file.size, "바이트입니다.")
