import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import os
import io

# Import untuk YOLOv8
from ultralytics import YOLO

# Import untuk DETR
from transformers import DetrForObjectDetection, DetrImageProcessor
import torch

# --- Konfigurasi Streamlit ---
st.set_page_config(
    page_title="Aplikasi Deteksi Manusia",
    page_icon="🚶‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Fungsi Pemuatan Model (Menggunakan Cache untuk Efisiensi) ---
@st.cache_resource
def load_yolov8s_model():
    """Memuat model YOLOv8s."""
    try:
        model = YOLO('yolov8s.pt') # Mengunduh jika belum ada
        st.success("Model YOLOv8s berhasil dimuat!")
        return model
    except Exception as e:
        st.error(f"Gagal memuat model YOLOv8s: {e}")
        return None

@st.cache_resource
def load_detr_model_and_processor():
    """Memuat model DETR dan image processor."""
    try:
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        st.success("Model DETR berhasil dimuat!")
        return processor, model
    except Exception as e:
        st.error(f"Gagal memuat model DETR: {e}. Pastikan koneksi internet Anda stabil.")
        st.warning("Jika error berlanjut, coba instal torch dengan dukungan CUDA (jika ada GPU).")
        return None, None

# Muat model saat aplikasi dimulai
yolov8s_model = load_yolov8s_model()
detr_processor, detr_model = load_detr_model_and_processor()

# Tentukan device (GPU jika tersedia, CPU jika tidak)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if detr_model:
    detr_model.to(device)

# --- Fungsi Deteksi ---
def detect_yolov8s(model, image, confidence_threshold):
    """
    Melakukan deteksi objek menggunakan YOLOv8s.
    Hanya mendeteksi 'person' (kelas 0 di COCO).
    """
    if model is None:
        return []

    results = model(image, conf=confidence_threshold)
    detections = []
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()

        for i in range(len(boxes)):
            # COCO class_id untuk 'person' adalah 0
            if classes[i] == 0:
                x1, y1, x2, y2 = map(int, boxes[i])
                score = scores[i]
                detections.append({
                    'box': [x1, y1, x2, y2],
                    'label': 'person',
                    'score': score
                })
    return detections

def detect_detr(processor, model, image, confidence_threshold):
    """
    Melakukan deteksi objek menggunakan DETR.
    Hanya mendeteksi 'person'.
    """
    if processor is None or model is None:
        return []

    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)

    # Mengubah output ke format yang mudah dibaca
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=confidence_threshold)[0]

    detections = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        # label mapping (COCO dataset)
        # 91 kelas, 'person' adalah kelas 1
        # Menggunakan COCO dataset labels, index 1 adalah 'person'
        # Pastikan mapping label sudah benar
        # (Anda mungkin perlu membuat dictionary mapping index ke nama kelas)
        # DetrImageProcessor.post_process_object_detection sudah memberikan label nama kelas
        
        # Contoh mapping label untuk DETR jika diperlukan
        # labels dari DETR adalah index COCO.
        # Person adalah index 0 di COCO dataset asli, atau 1 di HF transformers dataset
        # Cek: https://huggingface.co/facebook/detr-resnet-50/blob/main/config.json
        # id2label": {"0": "N/A", "1": "person", ...}
        
        if model.config.id2label[label.item()] == "person": # Pastikan ini sesuai dengan ID person
            x1, y1, x2, y2 = box.cpu().numpy().astype(int)
            detections.append({
                'box': [x1, y1, x2, y2],
                'label': 'person',
                'score': score.item()
            })
    return detections

def draw_boxes(image_np, detections):
    """Menggambar bounding box pada gambar NumPy array."""
    image_copy = image_np.copy()
    for det in detections:
        x1, y1, x2, y2 = det['box']
        label = det['label']
        score = det['score']

        cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label}: {score:.2f}"
        cv2.putText(image_copy, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return image_copy

# --- Sidebar ---
st.sidebar.header("Pengaturan Aplikasi")

detection_model_choice = st.sidebar.radio(
    "Pilih Model Deteksi:",
    ("YOLOv8s", "DETR")
)

confidence_threshold = st.sidebar.slider(
    "Threshold Keyakinan Deteksi:",
    min_value=0.0, max_value=1.0, value=0.5, step=0.05,
    help="Ambang batas kepercayaan untuk deteksi objek. Deteksi di bawah nilai ini akan diabaikan."
)

st.sidebar.markdown("---")
st.sidebar.header("Tentang Aplikasi")
st.sidebar.info(
    "Aplikasi prototipe ini dikembangkan untuk mendemonstrasikan perbandingan dua algoritma deteksi objek mutakhir: "
    "**YOLOv8s** (You Only Look Once versi 8 small) dan **DETR** (DEtection TRansformer). "
    "Fokus utama adalah deteksi manusia dalam gambar dan video."
)

st.sidebar.markdown("---")
st.sidebar.header("Identitas Pengembang")
st.sidebar.write("Nama: [Nama Anda/Tim Anda]")
st.sidebar.write("Institusi: [Institusi Anda]")
st.sidebar.write("Versi: 1.0.0")

st.sidebar.markdown("---")
st.sidebar.header("Cara Penggunaan")
st.sidebar.markdown("""
1.  **Pilih Model Deteksi** di sidebar.
2.  **Atur Threshold Keyakinan** di sidebar.
3.  **Unggah Gambar** atau **Video** di area utama.
4.  Klik tombol **"Deteksi dengan [Nama Model]"**.
5.  Lihat hasilnya dan perbandingan kinerja model.
""")


# --- Main Content ---
st.title("🚶‍♂️ Aplikasi Deteksi Manusia")
st.markdown("Aplikasi ini membandingkan kinerja **YOLOv8s** dan **DETR** untuk deteksi manusia.")

st.markdown("""
<style>
.stButton>button {
    background-color: #4CAF50; /* Green */
    color: white;
    padding: 10px 24px;
    border: none;
    border-radius: 8px;
    font-size: 16px;
    cursor: pointer;
    box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    transition: 0.3s;
}
.stButton>button:hover {
    background-color: #45a049;
    box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
}
.stSpinner > div > div {
    font-size: 1.2em;
    color: #4CAF50;
}
</style>
""", unsafe_allow_html=True)


col_upload, col_results = st.columns([1, 2])

with col_upload:
    st.header("Unggah Media")
    uploaded_file = st.file_uploader(
        "Pilih gambar atau video...",
        type=["jpg", "jpeg", "png", "mp4", "avi", "mov"],
        help="Unggah file gambar (JPG, JPEG, PNG) atau video (MP4, AVI, MOV) untuk dideteksi."
    )

    if uploaded_file is not None:
        file_type = uploaded_file.type
        st.subheader("File Asli")
        if "image" in file_type:
            image_pil = Image.open(uploaded_file)
            st.image(image_pil, caption="Gambar Asli", use_column_width=True)
            image_np = np.array(image_pil) # Convert PIL Image to NumPy array for OpenCV
            image_np_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB) # Streamlit displays RGB, OpenCV reads BGR

            if st.button(f"Deteksi dengan **{detection_model_choice}**"):
                with col_results:
                    st.subheader("Hasil Deteksi & Perbandingan")
                    st.write(f"Mendeteksi dengan **{detection_model_choice}**...")

                    start_time = time.time()
                    detections = []
                    num_detections = 0

                    if detection_model_choice == "YOLOv8s":
                        if yolov8s_model:
                            with st.spinner("Memproses gambar dengan YOLOv8s..."):
                                detections = detect_yolov8s(yolov8s_model, image_np_rgb, confidence_threshold)
                        else:
                            st.error("Model YOLOv8s tidak tersedia. Silakan cek pesan error saat pemuatan.")
                    elif detection_model_choice == "DETR":
                        if detr_processor and detr_model:
                            with st.spinner("Memproses gambar dengan DETR..."):
                                detections = detect_detr(detr_processor, detr_model, image_pil, confidence_threshold)
                        else:
                            st.error("Model DETR tidak tersedia. Silakan cek pesan error saat pemuatan.")

                    end_time = time.time()
                    inference_time = end_time - start_time
                    num_detections = len(detections)

                    if detections:
                        processed_image_np = draw_boxes(image_np_rgb, detections)
                        st.image(processed_image_np, caption=f"Hasil Deteksi ({detection_model_choice})", use_column_width=True)
                        st.success(f"Deteksi selesai! Ditemukan {num_detections} manusia.")
                    else:
                        st.warning("Tidak ada manusia terdeteksi.")

                    st.subheader("Metrik Kinerja")
                    st.markdown(f"""
                    | Metrik                  | Nilai                       |
                    | :---------------------- | :-------------------------- |
                    | Model Digunakan         | **{detection_model_choice}**|
                    | Waktu Inferensi         | **{inference_time:.4f} detik** |
                    | Jumlah Objek Terdeteksi | **{num_detections}** |
                    """)

                    st.markdown("---")
                    st.subheader("Perbandingan Model (Estimasi)")
                    st.info("Nilai ini adalah estimasi berdasarkan inferensi saat ini dan dapat bervariasi. Untuk perbandingan akurat, jalankan kedua model pada gambar yang sama secara terpisah.")
                    # Placeholder for comparison table
                    st.markdown("""
                    | Metrik             | YOLOv8s (Contoh Rata-rata) | DETR (Contoh Rata-rata)    |
                    | :----------------- | :------------------------- | :------------------------- |
                    | Waktu Inferensi    | Cepat (0.05 - 0.2 detik)   | Agak Lambat (0.1 - 0.5 detik) |
                    | Akurasi (Ideal)    | Tinggi                     | Sangat Tinggi              |
                    | Ukuran Model       | Relatif Kecil              | Agak Besar                 |
                    """)
                    st.markdown("""
                    **Catatan:** YOLOv8s umumnya lebih cepat untuk inferensi real-time, sementara DETR, meskipun lebih kompleks dan seringkali lebih akurat pada dataset besar, bisa lebih lambat karena arsitektur Transformer-nya.
                    """)


        elif "video" in file_type:
            st.video(uploaded_file)
            st.warning("Deteksi video akan segera tersedia! Fitur ini membutuhkan pemrosesan frame-by-frame yang lebih kompleks dan mungkin memakan waktu di Streamlit.")
            st.info("Untuk deteksi video, Anda dapat mengunduh file video dan memprosesnya secara offline menggunakan skrip Python yang terpisah.")
            # Placeholder for video processing logic
            # video_bytes = uploaded_file.read()
            # temp_video_path = "temp_video.mp4"
            # with open(temp_video_path, "wb") as f:
            #     f.write(video_bytes)

            # if st.button(f"Mulai Deteksi Video dengan {detection_model_choice}"):
            #     st.info("Memulai pemrosesan video. Ini bisa memakan waktu lama...")
            #     # Your video processing logic here
            #     # e.g., using OpenCV to read frames, detect, and save new video
            #     # Then display the processed video
            #     st.warning("Video processing feature is under development.")
            #     os.remove(temp_video_path)


    else:
        with col_results:
            st.info("Silakan unggah gambar atau video di sisi kiri untuk memulai deteksi.")

# --- Footer / Credits ---
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: grey; font-size: 0.9em;">
        Dibuat dengan ❤️ oleh [Nama Anda/Tim Anda] | Sumber Data Model: COCO Dataset | Model: <a href="https://docs.ultralytics.com/yolov8/" target="_blank">YOLOv8s (Ultralytics)</a>, <a href="https://huggingface.co/facebook/detr-resnet-50" target="_blank">DETR (Facebook AI)</a> | Powered by Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
