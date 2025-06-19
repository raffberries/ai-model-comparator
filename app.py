import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
import time
import pandas as pd

# Import MediaPipe
import mediapipe as mp

# Import MMPose
# Perlu memastikan mmpose dan mmdet terinstal dengan benar
try:
    from mmpose.apis import inference_detector, init_pose_model
    from mmdet.apis import inference_detector as det_inference_detector, init_detector as det_init_detector
    MMPose_AVAILABLE = True
except ImportError:
    MMPose_AVAILABLE = False
    st.warning("MMPose atau MMDetection tidak terinstal atau tidak dapat diimpor. Opsi MMPose mungkin tidak berfungsi.")

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="AI Model Comparator",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("✨ AI Model Comparator: Bandingkan Dua Model Sekaligus")
st.markdown("""
Aplikasi ini memungkinkan Anda untuk **membandingkan kinerja** dua model AI *pretrained* yang berbeda pada input yang sama.
Pilih jenis tugas, model yang ingin dibandingkan, lalu masukkan input Anda.
""")

st.markdown("---")

# Set device for model loading
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Utilitas untuk Visualisasi Pose (Diulang di sini agar app.py mandiri) ---
POSE_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
    (11, 12), (11, 13), (12, 14), (13, 15), (14, 16), (5,11), (6,12) # Tambahkan koneksi badan utama
]
KEYPOINT_COLOR = (0, 255, 0) # Hijau (BGR)
CONNECTION_COLOR = (0, 0, 255) # Merah (BGR)
RADIUS = 5 # Ukuran lingkaran keypoint
THICKNESS = 2 # Ketebalan garis koneksi

def draw_pose_on_image(image_np_bgr, keypoints, connections):
    output_image = image_np_bgr.copy()
    
    # Gambar koneksi
    for c in connections:
        p1_idx, p2_idx = c
        if len(keypoints) > p1_idx and len(keypoints) > p2_idx:
            p1 = keypoints[p1_idx]
            p2 = keypoints[p2_idx]
            if len(p1) > 2 and p1[2] > 0.3 and len(p2) > 2 and p2[2] > 0.3: # Check confidence
                cv2.line(output_image, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), CONNECTION_COLOR, THICKNESS)

    # Gambar keypoint
    for kp in keypoints:
        if len(kp) > 2 and kp[2] > 0.3: # Check confidence
            x, y, _ = kp
            cv2.circle(output_image, (int(x), int(y)), RADIUS, KEYPOINT_COLOR, -1)
    return output_image

def pil_to_cv2(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv2_image):
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

# --- Fungsi untuk Memuat Model dengan Caching ---
@st.cache_resource
def load_sentiment_model(model_path):
    return pipeline("sentiment-analysis", model=model_path)

@st.cache_resource
def load_mediapipe_pose():
    mp_pose = mp.solutions.pose
    pose_mp = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    return pose_mp

@st.cache_resource
def load_mmpose_model():
    if not MMPose_AVAILABLE:
        return None, None
    config_file = 'configs/body/2d_keypoint/topdown_heatmap/coco/hrnet_w32_coco_256x192.py'
    checkpoint_file = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth'
    
    # Pastikan file config ada di lingkungan Colab
    if not os.path.exists(config_file):
        st.error(f"File konfigurasi MMPose tidak ditemukan di {config_file}. Harap jalankan sel instalasi dan pengujian model di Colab.")
        return None, None

    # Load person detector for MMPose
    det_config = 'mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    det_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    
    if not os.path.exists(det_config):
        st.error(f"File konfigurasi MMDetection tidak ditemukan di {det_config}. Harap jalankan sel instalasi dan pengujian model di Colab.")
        return None, None

    person_detector = det_init_detector(det_config, det_checkpoint, device=device)
    pose_model_mmpose = init_pose_model(config_file, checkpoint_file, device=device)
    return person_detector, pose_model_mmpose

# --- Sidebar untuk Pilihan Utama (Jenis Tugas) ---
with st.sidebar:
    st.header("Konfigurasi Aplikasi")
    st.markdown("Pilih jenis tugas yang ingin Anda lakukan.")
    task_type = st.radio(
        "Jenis Tugas:",
        ("Analisis Sentimen (Teks)", "Estimasi Pose Manusia"),
        index=1 # Default ke Estimasi Pose
    )
    st.markdown("---")
    st.info("💡 **Tips:** Untuk estimasi pose, unggah gambar orang atau beberapa orang.")

# --- Bagian Utama Aplikasi ---
st.header("1. Pilih Model dan Unggah Input")

# Dictionary pilihan model
model_options_sentiment = {
    "DistilBERT Sentiment": "distilbert-base-uncased-finetuned-sst-2-english",
    "RoBERTa Twitter Sentiment": "cardiffnlp/twitter-roberta-base-sentiment"
}
model_options_pose = {
    "MediaPipe Pose": "MediaPipe Pose",
    "MMPose HRNet": "MMPose HRNet"
}

# Membuat kolom untuk pemilihan model
col1, col2 = st.columns(2)

# --- Logika untuk Analisis Sentimen (jika dipilih) ---
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
    user_input_text = st.text_area("Tulis teks di sini untuk dianalisis:", "I love Streamlit, it's so easy to use and powerful!", height=150)

    if st.button("Bandingkan Sentimen", use_container_width=True):
        if classifier1 and classifier2 and user_input_text:
            st.markdown("---")
            st.header("3. Hasil Perbandingan Sentimen")
            
            # Predict
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

# --- Logika untuk Estimasi Pose Manusia ---
elif task_type == "Estimasi Pose Manusia":
    with col1:
        st.subheader("Model 1 (Estimasi Pose)")
        model1_name_pose = st.selectbox(
            "Pilih model pertama:",
            list(model_options_pose.keys()),
            index=0,
            key="pose_model1"
        )
        img_classifier1 = None # Reset
        if model1_name_pose == "MediaPipe Pose":
            img_classifier1 = {"type": "mediapipe", "model": load_mediapipe_pose()}
            st.success(f"'{model1_name_pose}' siap!")
        elif model1_name_pose == "MMPose HRNet":
            if MMPose_AVAILABLE:
                person_detector, pose_model = load_mmpose_model()
                if person_detector and pose_model:
                    img_classifier1 = {"type": "mmpose", "detector": person_detector, "pose_model": pose_model}
                    st.success(f"'{model1_name_pose}' siap!")
                else:
                    st.error(f"Gagal memuat MMPose. Pastikan file config ada & instalasi benar.")
            else:
                st.error("MMPose tidak tersedia. Coba instal ulang sesuai petunjuk.")

    with col2:
        st.subheader("Model 2 (Estimasi Pose)")
        model2_name_pose = st.selectbox(
            "Pilih model kedua:",
            list(model_options_pose.keys()),
            index=1,
            key="pose_model2"
        )
        img_classifier2 = None # Reset
        if model2_name_pose == "MediaPipe Pose":
            img_classifier2 = {"type": "mediapipe", "model": load_mediapipe_pose()}
            st.success(f"'{model2_name_pose}' siap!")
        elif model2_name_pose == "MMPose HRNet":
            if MMPose_AVAILABLE:
                person_detector, pose_model = load_mmpose_model() # Cache akan menangani
                if person_detector and pose_model:
                    img_classifier2 = {"type": "mmpose", "detector": person_detector, "pose_model": pose_model}
                    st.success(f"'{model2_name_pose}' siap!")
                else:
                    st.error(f"Gagal memuat MMPose. Pastikan file config ada & instalasi benar.")
            else:
                st.error("MMPose tidak tersedia. Coba instal ulang sesuai petunjuk.")
    
    st.markdown("---")
    st.subheader("Unggah Gambar Anda")
    uploaded_file = st.file_uploader("Pilih gambar dari komputer Anda (gambar orang lebih baik):", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Gambar yang Diunggah', use_column_width=True)

        if st.button("Bandingkan Pose", use_container_width=True):
            if img_classifier1 and img_classifier2:
                st.markdown("---")
                st.header("3. Hasil Perbandingan Estimasi Pose")
                
                col_res1, col_res2 = st.columns(2)

                # Prediksi untuk Model 1
                with col_res1:
                    st.info(f"**{model1_name_pose}**")
                    start_time1 = time.time()
                    image_np_bgr = pil_to_cv2(image)
                    keypoints1 = []
                    
                    if img_classifier1["type"] == "mediapipe":
                        results = img_classifier1["model"].process(np.array(image))
                        if results.pose_landmarks:
                            for landmark in results.pose_landmarks.landmark:
                                h, w, _ = np.array(image).shape
                                keypoints1.append([landmark.x * w, landmark.y * h, landmark.visibility])
                    elif img_classifier1["type"] == "mmpose":
                        det_results = det_inference_detector(img_classifier1["detector"], image_np_bgr)
                        person_results = [person for person in det_results[0] if person[-1] > 0.5] # Confidence > 0.5
                        if person_results:
                            pose_results, _ = inference_detector(img_classifier1["pose_model"], image_np_bgr, det_bboxes=[person_results[0]])
                            if pose_results and pose_results[0].get('keypoints') is not None:
                                keypoints1 = pose_results[0]['keypoints'].tolist()
                    
                    inference_time1 = time.time() - start_time1
                    st.write(f"**Waktu Inferensi:** {inference_time1:.4f} detik")
                    st.write(f"**Jumlah Keypoint Terdeteksi:** {len(keypoints1)}")
                    
                    # Gambar pose pada gambar
                    drawn_image1 = draw_pose_on_image(image_np_bgr.copy(), keypoints1, POSE_CONNECTIONS)
                    st.image(cv2_to_pil(drawn_image1), caption=f"Pose oleh {model1_name_pose}", use_column_width=True)
                    with st.expander("Lihat Detail Keypoint JSON"):
                        st.json(keypoints1)

                # Prediksi untuk Model 2
                with col_res2:
                    st.info(f"**{model2_name_pose}**")
                    start_time2 = time.time()
                    image_np_bgr = pil_to_cv2(image)
                    keypoints2 = []

                    if img_classifier2["type"] == "mediapipe":
                        results = img_classifier2["model"].process(np.array(image))
                        if results.pose_landmarks:
                            for landmark in results.pose_landmarks.landmark:
                                h, w, _ = np.array(image).shape
                                keypoints2.append([landmark.x * w, landmark.y * h, landmark.visibility])
                    elif img_classifier2["type"] == "mmpose":
                        det_results = det_inference_detector(img_classifier2["detector"], image_np_bgr)
                        person_results = [person for person in det_results[0] if person[-1] > 0.5] # Confidence > 0.5
                        if person_results:
                            pose_results, _ = inference_detector(img_classifier2["pose_model"], image_np_bgr, det_bboxes=[person_results[0]])
                            if pose_results and pose_results[0].get('keypoints') is not None:
                                keypoints2 = pose_results[0]['keypoints'].tolist()
                    
                    inference_time2 = time.time() - start_time2
                    st.write(f"**Waktu Inferensi:** {inference_time2:.4f} detik")
                    st.write(f"**Jumlah Keypoint Terdeteksi:** {len(keypoints2)}")

                    # Gambar pose pada gambar
                    drawn_image2 = draw_pose_on_image(image_np_bgr.copy(), keypoints2, POSE_CONNECTIONS)
                    st.image(cv2_to_pil(drawn_image2), caption=f"Pose oleh {model2_name_pose}", use_column_width=True)
                    with st.expander("Lihat Detail Keypoint JSON"):
                        st.json(keypoints2)
            else:
                st.error("❌ Pastikan kedua model berhasil dimuat sebelum membandingkan. Periksa pesan error di atas.")
        else:
            st.info("👆 Unggah gambar dan klik tombol 'Bandingkan Pose'.")
    elif st.button("Bandingkan Pose"):
        st.warning("⚠️ Mohon unggah gambar terlebih dahulu untuk perbandingan.")

st.markdown("---")
st.markdown("Dibuat dengan ❤️ oleh Anda menggunakan Streamlit, MediaPipe, dan MMPose.")