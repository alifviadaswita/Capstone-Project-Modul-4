# ============================================
# CAPSTONE MODULE 4 - CONSTRUCTION SAFETY DETECTION
# Streamlit Application 
# ============================================

import streamlit as st
import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd
import io
import datetime
from typing import List, Dict

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
# LOAD YOLO MODEL (cached)
# ============================================
@st.cache_resource
def load_model(path: str = "models/best.pt"):
    try:
        model = YOLO(path)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None


# ============================================
# PROCESS IMAGE -> returns list of detections
# ============================================
def process_image(image_pil: Image.Image, model, conf_thres: float, iou_thres: float) -> List[Dict]:
    img_array = np.array(image_pil)  # RGB
    try:
        results = model.predict(
            source=img_array,
            conf=conf_thres,
            iou=iou_thres,
            verbose=False
        )
    except Exception as e:
        st.error(f"Error during model prediction: {e}")
        return []

    detections = []
    if len(results) > 0 and getattr(results[0], "boxes", None) is not None:
        boxes = results[0].boxes
        for i in range(len(boxes)):
            coords = boxes.xyxy[i].cpu().numpy().astype(int)
            x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
            conf = float(boxes.conf[i].cpu().numpy())
            cls_id = int(boxes.cls[i].cpu().numpy())
            cls_name = model.names.get(cls_id, str(cls_id))
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "class_name": cls_name
            })
    return detections


# ============================================
# DRAW DETECTIONS 
# ============================================
def draw_detections(image_rgb: np.ndarray, detections: List[Dict]) -> np.ndarray:
    img_bgr = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)

    color_map = {
        "helmet": (0, 255, 0),      # green
        "vest": (0, 165, 255),      # orange-ish (BGR)
        "no-helmet": (0, 0, 255),   # red
        "no-vest": (255, 0, 255),   # magenta
        "person": (255, 255, 255),  # white
    }

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["class_name"]
        conf = det["confidence"]
        color = color_map.get(label, (128, 128, 128))
        # draw bbox
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
        text = f"{label} ({conf:.2f})"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_bgr, (x1, max(y1 - th - 6, 0)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(img_bgr, text, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    img_rgb_out = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb_out


# ============================================
# IMAGE TO BYTES 
# ============================================
def image_to_bytes(img_rgb_array: np.ndarray) -> bytes:
    img_pil = Image.fromarray(img_rgb_array)
    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG")
    return buf.getvalue()


# ============================================
# Build summary dataframe from detections
# ============================================
def build_summary_df(detections: List[Dict]) -> pd.DataFrame:
    counts = {}
    for d in detections:
        cls = d["class_name"]
        counts[cls] = counts.get(cls, 0) + 1
    if len(counts) == 0:
        # ensure all known classes present with 0
        counts = {"helmet": 0, "no-helmet": 0, "vest": 0, "no-vest": 0, "person": 0}
    return pd.DataFrame(list(counts.items()), columns=["Class", "Count"])


# ============================================
# MAIN
# ============================================
def main():
    st.markdown("<h1 style='text-align: center;'>🦺 Construction Safety Detection System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Deteksi penggunaan alat keselamatan kerja (helm dan rompi)</p>", unsafe_allow_html=True)
    st.markdown("---")

    model = load_model()
    if model is None:
        st.stop()

    # ---------------------------
    # SIDEBAR
    # ---------------------------
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        conf_thres = st.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)
        iou_thres = st.slider("IoU Threshold", 0.1, 1.0, 0.45, 0.05)

        st.markdown("---")
        st.markdown("## 📌 Model Information")
        st.info(
            f"Model: YOLOv8 (file: models/best.pt)\n\n"
            f"Available classes (total {len(model.names)}):\n" +
            "\n".join([f"- {v}" for _, v in model.names.items()])
        )

        st.markdown("---")
        st.markdown("## ℹ️ About")
        st.write(
            "System ini mendeteksi pekerja konstruksi dan apakah mereka menggunakan alat keselamatan:\n"
            "- Deteksi objek (helm & rompi)\n- Analisis kepatuhan K3\n- Ekspor hasil (gambar / csv / json)"
        )

        st.markdown("---")
        st.markdown("## 🕒 History")
        if "history" not in st.session_state:
            st.session_state["history"] = []

        if len(st.session_state["history"]) == 0:
            st.info("Belum ada histori deteksi.")
        else:
            # show last 5
            for rec in reversed(st.session_state["history"][-5:]):
                st.markdown(f"📅 **{rec['timestamp']}** — {rec['summary']}")

        st.markdown("---")
        if st.button("🧹 Clear History"):
            st.session_state["history"] = []
            st.success("History dikosongkan.")

    # ---------------------------
    # UPLOAD AREA
    # ---------------------------
    uploaded_file = st.file_uploader("📤 Upload Gambar Lokasi Konstruksi", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image_pil = Image.open(uploaded_file).convert("RGB")
        except Exception as e:
            st.error(f"Gagal membuka file gambar: {e}")
            return

        st.subheader("📸 Preview Gambar")
        st.image(image_pil, use_container_width=True)

        width, height = image_pil.size
        st.caption(f"Resolusi: {width} x {height} px | Format: {uploaded_file.type}")

        start_detection = st.button("🚀 Mulai Deteksi")

        if start_detection:
            with st.spinner("Sedang memproses gambar..."):
                detections = process_image(image_pil, model, conf_thres, iou_thres)
                img_rgb = np.array(image_pil)  
                img_with_boxes = draw_detections(img_rgb, detections)

            col_orig, col_res = st.columns(2)
            with col_orig:
                st.subheader("📸 Original Image")
                st.image(image_pil, use_container_width=True)
            with col_res:
                st.subheader("🎯 Detection Results")
                st.image(img_with_boxes, use_container_width=True)

            st.markdown("---")
            st.subheader("📊 Detection Summary")

            df_summary = build_summary_df(detections)
            counts_map = dict(df_summary.values)
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Helmet", counts_map.get("helmet", 0))
            c2.metric("No Helmet", counts_map.get("no-helmet", 0))
            c3.metric("Vest", counts_map.get("vest", 0))
            c4.metric("No Vest", counts_map.get("no-vest", 0))
            c5.metric("Person", counts_map.get("person", 0))

            total_person = counts_map.get("person", 0)
            total_no_helmet = counts_map.get("no-helmet", 0)
            total_no_vest = counts_map.get("no-vest", 0)

            total_incomplete = max(total_no_helmet, total_no_vest)
            total_complete = max(total_person - total_incomplete, 0)

            st.markdown("### 🧍‍♂️ Kepatuhan Alat Keselamatan")

            colA, colB = st.columns(2)
            colA.metric("Pekerja Lengkap Alat", total_complete)
            colB.metric("Pekerja Tidak Lengkap", total_incomplete)

            if total_incomplete > 0:
                st.error(f"⚠️ Ada {total_incomplete} pekerja yang tidak lengkap alat keselamatannya!")
            else:
                st.success("✅  Semua pekerja sudah lengkap alat keselamatannya!")


            st.markdown("**Tabel Ringkasan**")
            st.dataframe(df_summary, use_container_width=True)

            positive_counts = {k: v for k, v in counts_map.items() if v > 0}
            if len(positive_counts) > 0:
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.pie(
                    positive_counts.values(),
                    labels=positive_counts.keys(),
                    autopct="%1.1f%%",
                    startangle=90
                )
                ax.set_title("Distribusi Kelas Deteksi (hanya >0)")
                st.pyplot(fig)
            else:
                st.info("Tidak ada objek terdeteksi untuk ditampilkan di diagram.")

            st.markdown("---")
            st.subheader("💾 Unduh Hasil Deteksi")

            try:
                img_bytes = image_to_bytes(img_with_boxes)
                st.download_button(
                    label="📸 Download Gambar Hasil",
                    data=img_bytes,
                    file_name="detection_result.jpg",
                    mime="image/jpeg"
                )
            except Exception as e:
                st.error(f"Gagal menyiapkan gambar untuk diunduh: {e}")

            try:
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
            except Exception as e:
                st.error(f"Gagal menyiapkan CSV/JSON untuk diunduh: {e}")

            summary_text = ", ".join([f"{row['Count']} {row['Class']}" for _, row in df_summary.iterrows()])
            st.session_state["history"].append({
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "summary": summary_text
            })

    else:
        st.info("👆 Silakan upload gambar untuk memulai.")

    st.markdown("---")
    st.markdown("<p style='text-align: center; color: gray;'>Capstone Project Module 4 - AI Engineer Training Program</p>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
