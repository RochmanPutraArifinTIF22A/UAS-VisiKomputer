# UAS-VisiKomputer
import streamlit as st
import cv2
import tempfile
import numpy as np

st.title("🎥 Deteksi Orang dalam Video")

uploaded_video = st.file_uploader("Unggah Video", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()

    # Inisialisasi detektor HOG (Histogram of Oriented Gradients)
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize untuk kecepatan
        frame = cv2.resize(frame, (640, 360))

        # Deteksi orang
        boxes, _ = hog.detectMultiScale(frame, winStride=(8, 8))

        # Gambar kotak
        for box in boxes:
            if len(box) == 4:
                x, y, w, h = box
            # Gambar kotak deteksi
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Tambahkan label di atas kotak
                cv2.putText(frame, "Manusia", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Tampilkan
        stframe.image(frame, channels="BGR", use_column_width=True)

    cap.release()
    st.success("🎉 Video selesai diproses.")

