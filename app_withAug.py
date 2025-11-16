import streamlit as st
import io
import datetime
from PIL import Image

import numpy as np
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(
    page_title="Construction Safety Detection System",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model(path="models/best_withAug.pt"):
    try:
        model = YOLO(path)
        return model
    except Exception as e:
        st.error("Error loading model. Pastikan file model ada di folder `models/`.\n\nDetail: " + str(e))
        return None

def process_image(image, model, conf_thres, iou_thres):
    img_array = np.array(image.convert("RGB"))
    results = model.predict(
        source=img_array,
        conf=conf_thres,
        iou=iou_thres,
        verbose=False
    )
    detections = []
    if len(results) > 0 and getattr(results[0], "boxes", None) is not None:
        boxes = results[0].boxes
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
            conf = float(boxes.conf[i].cpu().numpy())
            cls_id = int(boxes.cls[i].cpu().numpy())
            cls_name = model.names[cls_id]
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': conf,
                'class_name': cls_name
            })
    return detections, results

def draw_detections(image, detections):
    img_draw = image.copy()
    color_map = {
        'helmet': (0, 255, 0),
        'vest': (0, 255, 255),
        'no-helmet': (0, 0, 255),
        'no-vest': (255, 0, 0),
        'person': (255, 255, 255)
    }
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        label = det['class_name']
        conf = det['confidence']
        color = color_map.get(label, (128, 128, 128))
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img_draw,
            f"{label} ({conf:.2f})",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )
    return img_draw

def image_to_bytes(img_array):
    img = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

def main():
    st.markdown("<h1 style='text-align: center;'>🦺 Construction Safety Detection System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Deteksi penggunaan alat keselamatan kerja (helm dan rompi)</p>", unsafe_allow_html=True)
    st.markdown("---")

    model = load_model()
    if model is None:
        st.stop()

    with st.sidebar:
        st.header("⚙️ Settings")
        conf_thres = st.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)
        iou_thres = st.slider("IoU Threshold", 0.1, 1.0, 0.45, 0.05)
        st.info("Gunakan slider untuk menyesuaikan sensitivitas deteksi")
        st.markdown("---")
        st.subheader("🕒 History")
        if "history" in st.session_state and len(st.session_state["history"]) > 0:
            for record in reversed(st.session_state["history"][-5:]):
                st.markdown(f"📅 **{record['timestamp']}** — {record['summary']}")
        else:
            st.info("Belum ada histori deteksi.")

    uploaded_file = st.file_uploader("📤 Upload Gambar Lokasi Konstruksi", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📸 Original Image")
            st.image(image, use_column_width=True)

        with col2:
            st.subheader("🎯 Detection Results")
            with st.spinner("Sedang memproses gambar..."):
                detections, results = process_image(image, model, conf_thres, iou_thres)
                if len(results) > 0:
                    try:
                        plotted = results[0].plot()
                        st.image(plotted, use_column_width=True)
                        img_with_boxes = cv2.cvtColor(plotted, cv2.COLOR_RGB2BGR)
                    except Exception:
                        img_with_boxes = draw_detections(np.array(image), detections)
                        st.image(img_with_boxes, use_column_width=True)
                else:
                    img_with_boxes = draw_detections(np.array(image), detections)
                    st.image(img_with_boxes, use_column_width=True)

        # Summary
        st.markdown("---")
        st.subheader("📊 Detection Summary")
        class_counts = {}
        for det in detections:
            cls = det['class_name']
            class_counts[cls] = class_counts.get(cls, 0) + 1

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Helmet", class_counts.get('helmet', 0))
        c2.metric("No Helmet", class_counts.get('no-helmet', 0))
        c3.metric("Vest", class_counts.get('vest', 0))
        c4.metric("No Vest", class_counts.get('no-vest', 0))
        c5.metric("Person", class_counts.get('person', 0))

        incomplete_safety = class_counts.get('no-helmet', 0) + class_counts.get('no-vest', 0)
        if incomplete_safety > 0:
            st.warning(f"⚠️ {incomplete_safety} pekerja tidak lengkap alat keselamatannya!")
        else:
            st.success("✅ Semua pekerja sudah lengkap alat keselamatannya.")

        df_summary = pd.DataFrame(list(class_counts.items()), columns=["Class", "Count"])
        st.dataframe(df_summary, use_container_width=True)

        if len(class_counts) > 0:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie(class_counts.values(), labels=class_counts.keys(), autopct='%1.1f%%', startangle=90)
            ax.set_title("Distribusi Kelas Deteksi")
            st.pyplot(fig)

        # Download
        st.markdown("---")
        img_bytes = image_to_bytes(img_with_boxes)
        st.download_button("📸 Download Gambar Hasil", data=img_bytes, file_name="detection_result.jpg", mime="image/jpeg")
        csv_data = df_summary.to_csv(index=False).encode("utf-8")
        st.download_button("📑 Download Data CSV", data=csv_data, file_name="detection_summary.csv", mime="text/csv")
        json_data = df_summary.to_json(orient="records")
        st.download_button("📘 Download Data JSON", data=json_data, file_name="detection_summary.json", mime="application/json")

        if "history" not in st.session_state:
            st.session_state["history"] = []
        st.session_state["history"].append({
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "summary": f\"{class_counts.get('helmet', 0)} helmet, {class_counts.get('vest', 0)} vest, {class_counts.get('no-helmet', 0)} no-helmet, {class_counts.get('no-vest', 0)} no-vest\"
        })
    else:
        st.info("👆 Silakan upload gambar untuk memulai deteksi.")

    st.markdown("---")
    st.markdown("<p style='text-align: center; color: gray;'>Capstone Project Module 4 - AI Engineer Training Program</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
