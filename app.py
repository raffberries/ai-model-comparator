import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont # Tambahkan ImageDraw dan ImageFont
import torch
import time
import pandas as pd
import os

# Import Hugging Face for DETR pipelines
from transformers import pipeline

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="AI Model Comparator",
    page_icon="✨",
    layout="wide", # Menggunakan layout lebar untuk tampilan konten yang lebih baik
    initial_sidebar_state="expanded" # Sidebar dibuka secara default
)

# --- Judul dan Deskripsi Utama Aplikasi ---
st.title("✨ AI Model Comparator: Bandingkan Dua Model Sekaligus")
st.markdown("""
Aplikasi ini memungkinkan Anda untuk **membandingkan kinerja** dua model AI *pretrained* yang berbeda pada input yang sama.
Pilih jenis tugas, model yang ingin dibandingkan, lalu masukkan input Anda.
""")
st.markdown("---") # Garis pemisah visual

# Setel perangkat komputasi (GPU jika tersedia, jika tidak CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Utilitas Visualisasi Bounding Box (Pillow-only) ---
BBOX_COLOR_MODEL1_RGB = (255, 0, 0) # Merah (RGB) untuk model 1
BBOX_COLOR_MODEL2_RGB = (0, 0, 255) # Biru (RGB) untuk model 2
THICKNESS_BBOX = 2 # Ketebalan garis bounding box

def draw_bbox_on_image_pil(image_pil, detections, color_rgb):
    """
    Menggambar bounding box pada gambar PIL Image.
    Args:
        image_pil (PIL.Image.Image): Gambar PIL Image (RGB).
        detections (list): Daftar deteksi, setiap deteksi adalah dict dengan 'box', 'label', dan 'score'.
        color_rgb (tuple): Warna bounding box dalam format RGB (misal: (255, 0, 0) untuk merah).
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
            draw.rectangle([x1, y1, x2, y2], outline=color_rgb, width=THICKNESS_BBOX)
            
            text = f"{label_name}: {score:.2f}"
            text_bbox = draw.textbbox((x1, y1), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            # Gambar latar belakang teks
            draw.rectangle([x1, y1 - text_height - 5, x1 + text_width + 5, y1], fill=color_rgb)
            draw.text((x1 + 2, y1 - text_height - 3), text, fill=(255, 255, 255), font=font)
    return image_pil

# --- Fungsi untuk Memuat Model dengan Caching ---
@st.cache_resource
def load_sentiment_model(model_path):
    """Memuat model analisis sentimen dari Hugging Face."""
    return pipeline("sentiment-analysis", model=model_path)

@st.cache_resource
def load_object_detection_pipeline(model_path):
    """Memuat pipeline deteksi objek dari Hugging Face."""
    return pipeline("object-detection", model=model_path, device=0 if torch.cuda.is_available() else -1)

# --- Sidebar untuk Pilihan Utama (Jenis Tugas) ---
with st.sidebar:
    st.header("Konfigurasi Aplikasi")
    st.markdown("Pilih jenis tugas AI yang ingin Anda bandingkan:")
    task_type = st.radio(
        "Pilih Tugas:",
        ("Analisis Sentimen (Teks)", "Deteksi Objek"), # Ubah nama tugas
        index=1 # Default ke "Deteksi Objek"
    )
    st.markdown("---")
    st.info("💡 **Tips:** Unggah gambar dengan objek atau orang di dalamnya untuk hasil deteksi terbaik.")

# --- Bagian Utama Aplikasi ---
st.header("1. Pilih Model dan Unggah Input")

# Kamus pilihan model untuk setiap jenis tugas
model_options_sentiment = {
    "DistilBERT Sentiment": "distilbert-base-uncased-finetuned-sst-2-english",
    "RoBERTa Twitter Sentiment": "cardiffnlp/twitter-roberta-base-sentiment"
}
model_options_object_detection = { # Opsi untuk deteksi objek
    "DETR ResNet-50 (facebook/detr-resnet-50)": "facebook/detr-resnet-50",
    "Tiny DETR ResNet-50 (microsoft/resnet-50-tiny-detr)": "microsoft/resnet-50-tiny-detr"
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

# --- Logika untuk Deteksi Objek (jika 'Deteksi Objek' dipilih) ---
elif task_type == "Deteksi Objek":
    with col1:
        st.subheader("Model 1 (Detektor Objek)")
        model1_name_od = st.selectbox(
            "Pilih model pertama:",
            list(model_options_object_detection.keys()),
            index=0,
            key="od_model1" # Key unik
        )
        # Memuat pipeline deteksi objek
        od_pipeline1 = load_object_detection_pipeline(model_options_object_detection[model1_name_od])
        st.success(f"'{model1_name_od}' siap!")

    with col2:
        st.subheader("Model 2 (Detektor Objek)")
        model2_name_od = st.selectbox(
            "Pilih model kedua:",
            list(model_options_object_detection.keys()),
            index=1,
            key="od_model2" # Key unik
        )
        # Memuat pipeline deteksi objek
        od_pipeline2 = load_object_detection_pipeline(model_options_object_detection[model2_name_od])
        st.success(f"'{model2_name_od}' siap!")
    
    st.markdown("---")
    st.subheader("Unggah Gambar Anda")
    uploaded_file = st.file_uploader("Pilih gambar dari komputer Anda:", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Gambar yang Diunggah', use_column_width=True)

        if st.button("Bandingkan Deteksi Objek", use_container_width=True):
            if od_pipeline1 and od_pipeline2:
                st.markdown("---")
                st.header("3. Hasil Perbandingan Deteksi Objek")
                
                col_res1, col_res2 = st.columns(2)
                
                # Buat salinan gambar PIL untuk digambar oleh masing-masing model
                image_for_model1 = image.copy()
                image_for_model2 = image.copy()

                # --- Prediksi dan Tampilan untuk Model 1 ---
                with col_res1:
                    st.info(f"**{model1_name_od}**")
                    start_time1 = time.time()
                    
                    detections1 = od_pipeline1(image_for_model1)
                    drawn_image1 = draw_bbox_on_image_pil(image_for_model1, detections1, BBOX_COLOR_MODEL1_RGB) # Berikan warna
                    
                    inference_time1 = time.time() - start_time1
                    st.write(f"**Jumlah Objek Terdeteksi:** {len(detections1)}")
                    st.write(f"**Waktu Inferensi:** {inference_time1:.4f} detik")
                    st.image(drawn_image1, caption=f"Hasil dari {model1_name_od}", use_column_width=True)
                    with st.expander("Lihat Detail Deteksi JSON"):
                        st.json(detections1)
                    

                # --- Prediksi dan Tampilan untuk Model 2 ---
                with col_res2:
                    st.info(f"**{model2_name_od}**")
                    start_time2 = time.time()
                    
                    detections2 = od_pipeline2(image_for_model2)
                    drawn_image2 = draw_bbox_on_image_pil(image_for_model2, detections2, BBOX_COLOR_MODEL2_RGB) # Berikan warna berbeda
                    
                    inference_time2 = time.time() - start_time2
                    st.write(f"**Jumlah Objek Terdeteksi:** {len(detections2)}")
                    st.write(f"**Waktu Inferensi:** {inference_time2:.4f} detik")
                    st.image(drawn_image2, caption=f"Hasil dari {model2_name_od}", use_column_width=True)
                    with st.expander("Lihat Detail Deteksi JSON"):
                        st.json(detections2)
            else:
                st.error("❌ Pastikan kedua model berhasil dimuat sebelum membandingkan. Periksa pesan error di atas.")
        else:
            st.info("👆 Unggah gambar dan klik tombol 'Bandingkan Deteksi Objek'.")
    elif st.button("Bandingkan Deteksi Objek"):
        st.warning("⚠️ Mohon unggah gambar terlebih dahulu untuk perbandingan.")

st.markdown("---")
st.markdown("Dibuat dengan ❤️ oleh Anda menggunakan Streamlit dan Hugging Face Transformers.")
