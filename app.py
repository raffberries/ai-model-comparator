import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont # Import ImageDraw dan ImageFont untuk drawing
import torch
import time
import pandas as pd
import os # Untuk os.path.exists (walaupun tidak dipakai langsung di sini, bagus untuk kebiasaan)

# Import Hugging Face Transformers untuk model deteksi objek
from transformers import pipeline

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Human Detection App", # Judul halaman baru
    page_icon="🚶", # Ikon deteksi manusia
    layout="wide", # Menggunakan layout lebar
    initial_sidebar_state="expanded" # Sidebar terbuka
)

# --- Judul dan Deskripsi Aplikasi Utama ---
st.title("🚶 Aplikasi Deteksi Manusia") # Judul utama aplikasi
st.markdown("""
Aplikasi ini mendeteksi lokasi manusia (dan objek lain) dalam gambar menggunakan dua model AI yang berbeda.
Ini adalah langkah dasar dalam analisis pose.
""")
st.markdown("---") # Garis pemisah visual

# Setel perangkat komputasi (GPU jika tersedia, jika tidak CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Utilitas Visualisasi Bounding Box (Murni Pillow) ---
# Warna berbeda untuk setiap model agar mudah dibedakan
BBOX_COLOR_MODEL1_RGB = (255, 0, 0) # Merah (untuk Model 1)
BBOX_COLOR_MODEL2_RGB = (0, 0, 255) # Biru (untuk Model 2)
THICKNESS_BBOX = 2 # Ketebalan garis bounding box

def draw_bbox_on_image_pil(image_pil, detections, color_rgb):
    """
    Menggambar bounding box dan label pada gambar PIL Image.
    Args:
        image_pil (PIL.Image.Image): Gambar PIL Image (RGB).
        detections (list): Daftar deteksi, setiap deteksi adalah dict dengan 'box', 'label', dan 'score'.
        color_rgb (tuple): Warna bounding box dalam format RGB (misal: (255, 0, 0) untuk merah).
    Returns:
        PIL.Image.Image: Gambar dengan bounding box yang digambar.
    """
    draw = ImageDraw.Draw(image_pil)
    
    # Mencoba memuat font default, fallback jika gagal
    try:
        font = ImageFont.truetype("LiberationSans-Regular.ttf", 20) 
    except IOError:
        font = ImageFont.load_default() # Fallback ke font default Pillow

    for det in detections:
        box = det['box']
        label_name = det['label'] # Nama label (misal: 'person', 'car')
        score = det['score']

        if score > 0.7: # Gambar hanya jika confidence di atas ambang batas (dapat disesuaikan)
            x1, y1, x2, y2 = int(box['xmin']), int(box['ymin']), int(box['xmax']), int(box['ymax'])

            # Gambar persegi panjang bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color_rgb, width=THICKNESS_BBOX)
            
            # Persiapkan teks label
            text = f"{label_name}: {score:.2f}"
            text_bbox = draw.textbbox((x1, y1), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            # Gambar latar belakang teks agar lebih mudah dibaca
            draw.rectangle([x1, y1 - text_height - 5, x1 + text_width + 5, y1], fill=color_rgb)
            # Gambar teks label
            draw.text((x1 + 2, y1 - text_height - 3), text, fill=(255, 255, 255), font=font)
    return image_pil

# --- Fungsi untuk Memuat Model dengan Caching ---
# st.cache_resource akan memastikan model hanya diunduh dan dimuat sekali per sesi aplikasi
@st.cache_resource
def load_object_detection_pipeline(model_path):
    """
    Memuat pipeline deteksi objek dari Hugging Face.
    Args:
        model_path (str): ID model dari Hugging Face Hub (misal: "facebook/detr-resnet-50").
    Returns:
        transformers.Pipeline: Pipeline deteksi objek yang dimuat.
    """
    return pipeline("object-detection", model=model_path, device=0 if torch.cuda.is_available() else -1)

# --- Bagian Utama Aplikasi ---
st.header("1. Pilih Model Deteksi Manusia dan Unggah Gambar")

# Kamus pilihan model deteksi objek
model_options_object_detection = {
    "DETR ResNet-50 (facebook/detr-resnet-50)": "facebook/detr-resnet-50",
    "YOLOv8s (ultralytics/yolov8s)": "ultralytics/yolov8s" # Model YOLOv8s
}

# Membuat dua kolom untuk pemilihan model berdampingan
col1, col2 = st.columns(2)

# --- Pemilihan Model 1 ---
with col1:
    st.subheader("Model 1 (Detektor Manusia)")
    model1_name_od = st.selectbox(
        "Pilih model pertama:",
        list(model_options_object_detection.keys()),
        index=0,
        key="od_model1" # Key unik untuk widget Streamlit
    )
    # Memuat pipeline deteksi objek untuk Model 1
    od_pipeline1 = load_object_detection_pipeline(model_options_object_detection[model1_name_od])
    st.success(f"'{model1_name_od}' siap digunakan!")

# --- Pemilihan Model 2 ---
with col2:
    st.subheader("Model 2 (Detektor Manusia)")
    model2_name_od = st.selectbox(
        "Pilih model kedua:",
        list(model_options_object_detection.keys()),
        index=1,
        key="od_model2" # Key unik untuk widget Streamlit
    )
    # Memuat pipeline deteksi objek untuk Model 2
    od_pipeline2 = load_object_detection_pipeline(model_options_object_detection[model2_name_od])
    st.success(f"'{model2_name_od}' siap digunakan!")

st.markdown("---")
st.subheader("2. Unggah Gambar Anda")
# Widget file uploader untuk pengguna mengunggah gambar
uploaded_file = st.file_uploader("Pilih gambar dari komputer Anda (gambar orang lebih baik untuk deteksi manusia):", type=["jpg", "jpeg", "png", "webp"])

# --- PENANGANAN LOGIKA SETELAH FILE DIUNGGAH DAN TOMBOL DIKLIK ---
# Ini adalah blok utama yang berisi logic setelah gambar diunggah dan tombol diklik.
if uploaded_file is not None: # Mengecek apakah ada file yang diunggah
    image = Image.open(uploaded_file).convert('RGB') # Buka dan konversi gambar ke format RGB
    st.image(image, caption='Gambar yang Diunggah', use_column_width=True) # Tampilkan gambar yang diunggah

    # Tombol untuk memulai deteksi, hanya muncul setelah gambar diunggah
    if st.button("Mulai Deteksi Manusia", use_container_width=True): # Mengecek apakah tombol diklik
        if od_pipeline1 and od_pipeline2: # Pastikan kedua model berhasil dimuat
            st.markdown("---")
            st.header("3. Hasil Deteksi Manusia")
            
            # Membuat dua kolom untuk menampilkan hasil berdampingan
            col_res1, col_res2 = st.columns(2)
            
            # Buat salinan gambar PIL untuk digambar oleh masing-masing model
            # Ini penting agar kedua model menggambar di atas gambar asli yang sama
            image_for_model1 = image.copy()
            image_for_model2 = image.copy()

            # --- Deteksi dan Tampilan untuk Model 1 ---
            with col_res1:
                st.info(f"**{model1_name_od}**")
                start_time1 = time.time() # Mulai hitung waktu inferensi
                
                detections1 = od_pipeline1(image_for_model1) # Lakukan deteksi
                # Gambar bounding box pada salinan gambar untuk Model 1
                drawn_image1 = draw_bbox_on_image_pil(image_for_model1, detections1, BBOX_COLOR_MODEL1_RGB) 
                
                inference_time1 = time.time() - start_time1 # Selesai hitung waktu
                
                # Filter hanya deteksi 'person' untuk jumlah yang relevan
                person_count1 = sum(1 for det in detections1 if det['label'] == 'person')

                st.write(f"**Manusia Terdeteksi:** {person_count1} (Total Objek: {len(detections1)})")
                st.write(f"**Waktu Inferensi:** {inference_time1:.4f} detik")
                # Tampilkan gambar hasil
                st.image(drawn_image1, caption=f"Hasil dari {model1_name_od}", use_column_width=True)
                with st.expander("Lihat Detail Deteksi JSON"): # Expander untuk detail JSON
                    st.json(detections1)
                

            # --- Deteksi dan Tampilan untuk Model 2 ---
            with col_res2:
                st.info(f"**{model2_name_od}**")
                start_time2 = time.time() # Mulai hitung waktu inferensi
                
                detections2 = od_pipeline2(image_for_model2) # Lakukan deteksi
                # Gambar bounding box pada salinan gambar untuk Model 2
                drawn_image2 = draw_bbox_on_image_pil(image_for_model2, detections2, BBOX_COLOR_MODEL2_RGB) 
                
                inference_time2 = time.time() - start_time2 # Selesai hitung waktu
                
                # Filter hanya deteksi 'person' untuk jumlah yang relevan
                person_count2 = sum(1 for det in detections2 if det['label'] == 'person')

                st.write(f"**Manusia Terdeteksi:** {person_count2} (Total Objek: {len(detections2)})")
                st.write(f"**Waktu Inferensi:** {inference_time2:.4f} detik")
                # Tampilkan gambar hasil
                st.image(drawn_image2, caption=f"Hasil dari {model2_name_od}", use_column_width=True)
                with st.expander("Lihat Detail Deteksi JSON"): # Expander untuk detail JSON
                    st.json(detections2)
            else: # Ini adalah else untuk 'if od_pipeline1 and od_pipeline2:'
                st.error("❌ Pastikan kedua model berhasil dimuat sebelum mendeteksi. Periksa pesan error di atas.")
        else: # Ini adalah else untuk 'if st.button("Mulai Deteksi Manusia", use_container_width=True):' ketika tombol diklik tetapi models tidak siap
            st.info("👆 Klik tombol 'Mulai Deteksi Manusia' di atas untuk memulai deteksi.")

# Ini adalah else untuk 'if uploaded_file is not None:'
else: # Ini yang menangani kasus tidak ada file yang diunggah
    st.info("👆 Unggah gambar untuk memulai deteksi manusia.")
    # Opsional: Jika Anda ingin tombol "Mulai Deteksi Manusia" selalu ada bahkan tanpa file diunggah
    # Anda bisa menempatkan st.button di sini, tapi akan memicu warning jika belum ada file.
    # Misalnya:
    # if st.button("Mulai Deteksi Manusia", use_container_width=True):
    #     st.warning("⚠️ Mohon unggah gambar terlebih dahulu untuk deteksi.")

st.markdown("---")
st.markdown("Dibuat dengan ❤️ oleh Anda menggunakan Streamlit dan Hugging Face Transformers.")
