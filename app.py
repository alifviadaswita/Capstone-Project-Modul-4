# ============================================
# CAPSTONE MODULE 4 - CONSTRUCTION SAFETY DETECTION
# Streamlit Application 
# ============================================

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd
import io
import datetime

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Construction Safety Detection System",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# LOAD YOLO MODEL
# ============================================
@st.cache_resource
def load_model():
    try:
        model = YOLO('best.pt')  
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


# ============================================
# PROCESS IMAGE
# ============================================
def process_image(image, model, conf_thres, iou_thres):
    img_array = np.array(image)
    results = model.predict(
        source=img_array,
        conf=conf_thres,
        iou=iou_thres,
        verbose=False
    )
    detections = []
    if len(results) > 0 and results[0].boxes is not None:
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
    return detections


# ============================================
# DRAW DETECTIONS
# ============================================
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


# ============================================
# CONVERT IMAGE 
# ============================================
def image_to_bytes(img_array):
    img = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ============================================
# MAIN APPLICATION
# ============================================
def main():
    st.markdown("<h1 style='text-align: center;'>🦺 Construction Safety Detection System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Deteksi penggunaan alat keselamatan kerja (helm dan rompi)</p>", unsafe_allow_html=True)
    st.markdown("---")

    model = load_model()
    if model is None:
        st.stop()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        conf_thres = st.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)
        iou_thres = st.slider("IoU Threshold", 0.1, 1.0, 0.45, 0.05)
        st.info("Gunakan slider untuk menyesuaikan sensitivitas deteksi")

        st.markdown("---")
        st.subheader("🕒 History")
        if "history" in st.session_state and len(st.session_state["history"]) > 0:
            for record in reversed(st.session_state["history"][-5:]):  # tampilkan 5 terakhir
                st.markdown(f"📅 **{record['timestamp']}** — {record['summary']}")
        else:
            st.info("Belum ada histori deteksi.")

    # Upload image
    uploaded_file = st.file_uploader("📤 Upload Gambar Lokasi Konstruksi", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📸 Original Image")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("🎯 Detection Results")

            with st.spinner("Sedang memproses gambar..."):
                detections = process_image(image, model, conf_thres, iou_thres)
                img_array = np.array(image)
                img_with_boxes = draw_detections(img_array, detections)
                st.image(img_with_boxes, use_container_width=True)

        # Hasil
        st.markdown("---")
        st.subheader("📊 Detection Summary")

        class_counts = {}
        for det in detections:
            cls = det['class_name']
            class_counts[cls] = class_counts.get(cls, 0) + 1

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Helmet", class_counts.get('helmet', 0))
        col2.metric("No Helmet", class_counts.get('no-helmet', 0))
        col3.metric("Vest", class_counts.get('vest', 0))
        col4.metric("No Vest", class_counts.get('no-vest', 0))
        col5.metric("Person", class_counts.get('person', 0))

        incomplete_safety = class_counts.get('no-helmet', 0) + class_counts.get('no-vest', 0)
        if incomplete_safety > 0:
            st.warning(f"⚠️ {incomplete_safety} pekerja tidak lengkap alat keselamatannya!")
        else:
            st.success("✅ Semua pekerja sudah lengkap alat keselamatannya.")

        df_summary = pd.DataFrame(list(class_counts.items()), columns=["Class", "Count"])
        st.dataframe(df_summary, use_container_width=True)

        # Pie chart
        if len(class_counts) > 0:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie(
                class_counts.values(),
                labels=class_counts.keys(),
                autopct='%1.1f%%',
                startangle=90,
                colors=plt.cm.Set3.colors
            )
            ax.set_title("Distribusi Kelas Deteksi")
            st.pyplot(fig)

        # ============================================
        # DOWNLOAD SECTION
        # ============================================
        st.markdown("---")
        st.subheader("💾 Unduh Hasil Deteksi")

        img_bytes = image_to_bytes(img_with_boxes)
        st.download_button(
            label="📸 Download Gambar Hasil",
            data=img_bytes,
            file_name="detection_result.jpg",
            mime="image/jpeg"
        )

        csv_data = df_summary.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📑 Download Data CSV",
            data=csv_data,
            file_name="detection_summary.csv",
            mime="text/csv"
        )

        json_data = df_summary.to_json(orient="records")
        st.download_button(
            label="📘 Download Data JSON",
            data=json_data,
            file_name="detection_summary.json",
            mime="application/json"
        )

        # ============================================
        # HISTORY SECTION
        # ============================================
        if "history" not in st.session_state:
            st.session_state["history"] = []

        st.session_state["history"].append({
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "summary": f"{class_counts.get('helmet', 0)} helmet, {class_counts.get('vest', 0)} vest, "
                       f"{class_counts.get('no-helmet', 0)} no-helmet, {class_counts.get('no-vest', 0)} no-vest"
        })

    else:
        st.info("👆 Silakan upload gambar untuk memulai deteksi.")

    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: gray;'>Capstone Project Module 4 - AI Engineer Training Program</p>",
        unsafe_allow_html=True
    )


# ============================================
# RUN APP
# ============================================
if __name__ == "__main__":
    main()
