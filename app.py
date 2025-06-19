import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont # Tambahkan ImageDraw dan ImageFont
import torch
import time
import pandas as pd
import os

# Import MediaPipe
import mediapipe as mp

# Import Hugging Face for DETR
from transformers import pipeline

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="AI Model Comparator",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Judul dan Deskripsi Utama Aplikasi ---
st.title("✨ AI Model Comparator: Bandingkan Dua Model Sekaligus")
st.markdown("""
Aplikasi ini memungkinkan Anda untuk **membandingkan kinerja** dua model AI *pretrained* yang berbeda pada input yang sama.
Pilih jenis tugas, model yang ingin dibandingkan, lalu masukkan input Anda.
""")
st.markdown("---")

# Setel perangkat komputasi (GPU jika tersedia, jika tidak CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Utilitas Visualisasi Pose & Bounding Box (TANPA OpenCV) ---

# Definisi koneksi keypoint untuk pose manusia (berdasarkan COCO dataset)
# MediaPipe menyediakan ini secara internal, tapi ini untuk referensi menggambar manual
POSE_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
    (11, 12), (11, 13), (12, 14), (13, 15), (14, 16),
    (5, 11), (6, 12) # Koneksi bahu ke pinggul
]
KEYPOINT_COLOR_POSE_RGB = (0, 255, 0) # Hijau (RGB)
CONNECTION_COLOR_POSE_RGB = (255, 0, 0) # Merah (RGB)
RADIUS_POSE = 5 # Radius lingkaran keypoint
THICKNESS_POSE = 2 # Ketebalan garis koneksi

def draw_pose_on_image_pil(image_pil, keypoints, connections):
    """
    Menggambar keypoint dan koneksi pose pada gambar PIL Image.
    Args:
        image_pil (PIL.Image.Image): Gambar PIL Image (RGB).
        keypoints (list): Daftar keypoint, setiap keypoint adalah [x, y, confidence].
        connections (list): Daftar tuple (idx1, idx2) yang menunjukkan koneksi antar keypoint.
    Returns:
        PIL.Image.Image: Gambar dengan pose yang digambar.
    """
    draw = ImageDraw.Draw(image_pil)
    
    # Gambar koneksi antar keypoint
    for c in connections:
        p1_idx, p2_idx = c
        if (0 <= p1_idx < len(keypoints) and 0 <= p2_idx < len(keypoints)):
            p1 = keypoints[p1_idx]
            p2 = keypoints[p2_idx]
            if len(p1) > 2 and p1[2] > 0.3 and len(p2) > 2 and p2[2] > 0.3:
                draw.line([(p1[0], p1[1]), (p2[0], p2[1])], fill=CONNECTION_COLOR_POSE_RGB, width=THICKNESS_POSE)
    
    # Gambar setiap keypoint
    for kp in keypoints:
        if len(kp) > 2 and kp[2] > 0.3:
            x, y, _ = kp
            # Gambar lingkaran keypoint
            draw.ellipse([x - RADIUS_POSE, y - RADIUS_POSE, x + RADIUS_POSE, y + RADIUS_POSE], fill=KEYPOINT_COLOR_POSE_RGB)
    
    return image_pil

# Definisi warna dan ketebalan untuk Bounding Box (DETR)
BBOX_COLOR_DETR_RGB = (0, 0, 255) # Biru (RGB)
THICKNESS_BBOX = 2 # Ketebalan garis bounding box

def draw_bbox_on_image_pil(image_pil, detections):
    """
    Menggambar bounding box pada gambar PIL Image.
    Args:
        image_pil (PIL.Image.Image): Gambar PIL Image (RGB).
        detections (list): Daftar deteksi, setiap deteksi adalah dict dengan 'box' (xmin, ymin, xmax, ymax), 'label', dan 'score'.
    Returns:
        PIL.Image.Image: Gambar dengan bounding box yang digambar.
    """
    draw = ImageDraw.Draw(image_pil)
    
    # Coba memuat font default, fallback jika gagal (Streamlit Cloud mungkin tidak punya banyak font)
    try:
        font = ImageFont.truetype("LiberationSans-Regular.ttf", 20) # Font umum di Linux
    except IOError:
        font = ImageFont.load_default() # Fallback ke font default Pillow

    for det in detections:
        box = det['box']
        label_name = det['label']
        score = det['score']

        if score > 0.7: # Gambar hanya jika confidence di atas ambang batas
            x1, y1, x2, y2 = int(box['xmin']), int(box['ymin']), int(box['xmax']), int(box['ymax'])

            # Gambar persegi panjang bounding box
            draw.rectangle([x1, y1, x2, y2], outline=BBOX_COLOR_DETR_RGB, width=THICKNESS_BBOX)
            
            text = f"{label_name}: {score:.2f}"
            text_bbox = draw.textbbox((x1, y1), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            # Gambar latar belakang teks
            draw.rectangle([x1, y1 - text_height - 5, x1 + text_width + 5, y1], fill=BBOX_COLOR_DETR_RGB)
            draw.text((x1 + 2, y1 - text_height - 3), text, fill=(255, 255, 255), font=font)
    return image_pil

# --- Fungsi untuk Memuat Model dengan Caching ---
@st.cache_resource
def load_sentiment_model(model_path):
    return pipeline("sentiment-analysis", model=model_path)

@st.cache_resource
def load_mediapipe_pose_model():
    mp_pose = mp.solutions.pose
    pose_mp = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    return pose_mp

@st.cache_resource
def load_detr_pipeline():
    detr_pipe = pipeline("object-detection", model="facebook/detr-resnet-50", device=0 if torch.cuda.is_available() else -1)
    return detr_pipe

# --- Sidebar untuk Pilihan Utama (Jenis Tugas) ---
with st.sidebar:
    st.header("Konfigurasi Aplikasi")
    st.markdown("Pilih jenis tugas AI yang ingin Anda bandingkan:")
    task_type = st.radio(
        "Pilih Tugas:",
        ("Analisis Sentimen (Teks)", "Estimasi Pose & Deteksi Objek"),
        index=1 # Default ke "Estimasi Pose & Deteksi Objek"
    )
    st.markdown("---")
    st.info("💡 **Tips:** Untuk estimasi pose, unggah gambar yang jelas menunjukkan orang atau beberapa orang.")

# --- Bagian Utama Aplikasi ---
st.header("1. Pilih Model dan Unggah Input")

# Kamus pilihan model untuk setiap jenis tugas
model_options_sentiment = {
    "DistilBERT Sentiment": "distilbert-base-uncased-finetuned-sst-2-english",
    "RoBERTa Twitter Sentiment": "cardiffnlp/twitter-roberta-base-sentiment"
}
model_options_vision = {
    "MediaPipe Pose": "MediaPipe Pose",
    "DETR Object Detector": "DETR Object Detector"
}

# Membuat dua kolom untuk pemilihan model berdampingan
col1, col2 = st.columns(2)

# --- Logika untuk Analisis Sentimen (jika 'Analisis Sentimen (Teks)' dipilih) ---
if task_type == "Analisis Sentimen (Teks)":
    with col1:
        st.subheader("Model 1 (Sentimen)")
        model1_name = st.selectbox(
            "Pilih model pertama:",
            list(model_options_sentiment.keys()),
            index=0,
            key="sentiment_model1"
        )
        model1_path = model_options_sentiment[model1_name]
        try:
            classifier1 = load_sentiment_model(model1_path)
            st.success(f"'{model1_name}' siap!")
        except Exception as e:
            st.error(f"Gagal memuat model '{model1_name}': {e}")
            classifier1 = None

    with col2:
        st.subheader("Model 2 (Sentimen)")
        model2_name = st.selectbox(
            "Pilih model kedua:",
            list(model_options_sentiment.keys()),
            index=1,
            key="sentiment_model2"
        )
        model2_path = model_options_sentiment[model2_name]
        try:
            classifier2 = load_sentiment_model(model2_path)
            st.success(f"'{model2_name}' siap!")
        except Exception as e:
            st.error(f"Gagal memuat model '{model2_name}': {e}")
            classifier2 = None

    st.markdown("---")
    st.subheader("Masukkan Teks Anda")
    user_input_text = st.text_area("Tulis teks di sini untuk dianalisis:", "Saya sangat suka Streamlit, sangat mudah digunakan dan powerful!", height=150)

    if st.button("Bandingkan Sentimen", use_container_width=True):
        if classifier1 and classifier2 and user_input_text:
            st.markdown("---")
            st.header("3. Hasil Perbandingan Sentimen")
            
            res1 = classifier1(user_input_text)[0]
            res2 = classifier2(user_input_text)[0]

            col_res1, col_res2 = st.columns(2)

            with col_res1:
                st.info(f"**{model1_name}**")
                st.metric(label="Sentimen", value=res1['label'].capitalize())
                st.progress(res1['score'], text=f"Kepercayaan: {res1['score']:.2f}")
                with st.expander("Lihat Detail JSON"):
                    st.json(res1)

            with col_res2:
                st.info(f"**{model2_name}**")
                st.metric(label="Sentimen", value=res2['label'].capitalize())
                st.progress(res2['score'], text=f"Kepercayaan: {res2['score']:.2f}")
                with st.expander("Lihat Detail JSON"):
                    st.json(res2)
        elif not user_input_text:
            st.warning("☝️ Mohon masukkan teks untuk perbandingan.")
        else:
            st.error("❌ Pastikan kedua model berhasil dimuat sebelum membandingkan.")

# --- Logika untuk Estimasi Pose & Deteksi Objek (jika 'Estimasi Pose & Deteksi Objek' dipilih) ---
elif task_type == "Estimasi Pose & Deteksi Objek":
    with col1:
        st.subheader("Model 1 (Visi Komputer)")
        model1_name_vision = st.selectbox(
            "Pilih model pertama:",
            list(model_options_vision.keys()),
            index=0,
            key="vision_model1"
        )
        model1_instance = None
        
        if model1_name_vision == "MediaPipe Pose":
            model1_instance = {"type": "mediapipe", "model": load_mediapipe_pose_model()}
            st.success(f"'{model1_name_vision}' siap!")
        elif model1_name_vision == "DETR Object Detector":
            detr_pipe_instance = load_detr_pipeline()
            model1_instance = {"type": "detr", "pipeline": detr_pipe_instance}
            st.success(f"'{model1_name_vision}' siap!")
        else:
            st.error("Model tidak dikenali atau gagal dimuat.")
            model1_instance = None


    with col2:
        st.subheader("Model 2 (Visi Komputer)")
        model2_name_vision = st.selectbox(
            "Pilih model kedua:",
            list(model_options_vision.keys()),
            index=1,
            key="vision_model2"
        )
        model2_instance = None

        if model2_name_vision == "MediaPipe Pose":
            model2_instance = {"type": "mediapipe", "model": load_mediapipe_pose_model()}
            st.success(f"'{model2_name_vision}' siap!")
        elif model2_name_vision == "DETR Object Detector":
            detr_pipe_instance = load_detr_pipeline()
            model2_instance = {"type": "detr", "pipeline": detr_pipe_instance}
            st.success(f"'{model2_name_vision}' siap!")
        else:
            st.error("Model tidak dikenali atau gagal dimuat.")
            model2_instance = None
    
    st.markdown("---")
    st.subheader("Unggah Gambar Anda")
    uploaded_file = st.file_uploader("Pilih gambar dari komputer Anda:", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Gambar yang Diunggah', use_column_width=True)

        if st.button("Bandingkan Visi Komputer", use_container_width=True):
            if model1_instance and model2_instance:
                st.markdown("---")
                st.header("3. Hasil Perbandingan Visi Komputer")
                
                col_res1, col_res2 = st.columns(2)
                
                # Buat salinan gambar PIL untuk digambar oleh masing-masing model
                image_for_model1 = image.copy()
                image_for_model2 = image.copy()

                # --- Prediksi dan Tampilan untuk Model 1 ---
                with col_res1:
                    st.info(f"**{model1_name_vision}**")
                    start_time1 = time.time()
                    
                    if model1_instance["type"] == "mediapipe":
                        results = model1_instance["model"].process(np.array(image_for_model1))
                        keypoints1 = []
                        if results.pose_landmarks:
                            for landmark in results.pose_landmarks.landmark:
                                h, w, _ = np.array(image_for_model1).shape
                                keypoints1.append([landmark.x * w, landmark.y * h, landmark.visibility])
                        drawn_image1 = draw_pose_on_image_pil(image_for_model1, keypoints1, POSE_CONNECTIONS)
                        st.write(f"**Jumlah Keypoint Terdeteksi:** {len(keypoints1)}")
                        with st.expander("Lihat Detail Keypoint JSON"):
                            st.json(keypoints1)
                    elif model1_instance["type"] == "detr":
                        detections1 = model1_instance["pipeline"](image_for_model1)
                        drawn_image1 = draw_bbox_on_image_pil(image_for_model1, detections1)
                        st.write(f"**Jumlah Objek Terdeteksi:** {len(detections1)}")
                        with st.expander("Lihat Detail Deteksi JSON"):
                            st.json(detections1)
                    
                    inference_time1 = time.time() - start_time1
                    st.write(f"**Waktu Inferensi:** {inference_time1:.4f} detik")
                    st.image(drawn_image1, caption=f"Hasil dari {model1_name_vision}", use_column_width=True)
                    

                # --- Prediksi dan Tampilan untuk Model 2 ---
                with col_res2:
                    st.info(f"**{model2_name_vision}**")
                    start_time2 = time.time()
                    
                    if model2_instance["type"] == "mediapipe":
                        results = model2_instance["model"].process(np.array(image_for_model2))
                        keypoints2 = []
                        if results.pose_landmarks:
                            for landmark in results.pose_landmarks.landmark:
                                h, w, _ = np.array(image_for_model2).shape
                                keypoints2.append([landmark.x * w, landmark.y * h, landmark.visibility])
                        drawn_image2 = draw_pose_on_image_pil(image_for_model2, keypoints2, POSE_CONNECTIONS)
                        st.write(f"**Jumlah Keypoint Terdeteksi:** {len(keypoints2)}")
                        with st.expander("Lihat Detail Keypoint JSON"):
                            st.json(keypoints2)
                    elif model2_instance["type"] == "detr":
                        detections2 = model2_instance["pipeline"](image_for_model2)
                        drawn_image2 = draw_bbox_on_image_pil(image_for_model2, detections2)
                        st.write(f"**Jumlah Objek Terdeteksi:** {len(detections2)}")
                        with st.expander("Lihat Detail Deteksi JSON"):
                            st.json(detections2)
                    
                    inference_time2 = time.time() - start_time2
                    st.write(f"**Waktu Inferensi:** {inference_time2:.4f} detik")
                    st.image(drawn_image2, caption=f"Hasil dari {model2_name_vision}", use_column_width=True)
            else:
                st.error("❌ Pastikan kedua model berhasil dimuat sebelum membandingkan. Periksa pesan error di atas.")
        else:
            st.info("👆 Unggah gambar dan klik tombol 'Bandingkan Visi Komputer'.")
    elif st.button("Bandingkan Visi Komputer"):
        st.warning("⚠️ Mohon unggah gambar terlebih dahulu untuk perbandingan.")

st.markdown("---")
st.markdown("Dibuat dengan ❤️ oleh Anda menggunakan Streamlit, MediaPipe, dan Hugging Face Transformers.")
