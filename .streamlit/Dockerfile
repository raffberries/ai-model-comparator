# Dockerfile

# Menggunakan base image Python yang stabil
# Pilih versi Python yang sama dengan yang Anda gunakan di requirements.txt (misal, Python 3.9)
FROM python:3.9-slim-buster

# Instal dependensi sistem yang dibutuhkan oleh OpenCV
# libgl1: Pustaka OpenGL utama
# libglib2.0-0: Dependensi umum untuk banyak aplikasi GTK/GNOME
# libsm6, libxrender1, libxext6: Dependensi untuk sistem X Window (sering dibutuhkan oleh highgui OpenCV)
# libfontconfig1, libharfbuzz0b, libfreetype6: Dependensi font, seringkali diperlukan oleh OpenCV atau Matplotlib
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libfontconfig1 \
    libharfbuzz0b \
    libfreetype6 \
    && rm -rf /var/lib/apt/lists/*

# Buat direktori kerja di dalam container
WORKDIR /app

# Salin file requirements.txt dan install semua pustaka Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh kode aplikasi Streamlit Anda ke dalam container
# Pastikan app.py berada di root direktori yang disalin
COPY . .

# Perintah untuk menjalankan aplikasi Streamlit
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
