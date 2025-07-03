import os
import cv2
import pickle
import numpy as np
from PIL import Image
from keras_facenet import FaceNet
from numpy import asarray, expand_dims

# --- Inisialisasi Model ---
HaarCascade = cv2.CascadeClassifier(cv2.samples.findFile(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'))
MyFaceNet = FaceNet()

# --- Direktori Dataset ---
# Jika ingin menggunakan code ini struktur folder dataset harus seperti ini:
# --- Direktori Dataset ---
# Folder utama untuk dataset gambar wajah yang akan digunakan untuk pembuatan embeddings.
# Setiap subfolder di dalam 'dataset' akan mewakili satu orang, misalnya:
# dataset/
# ├── Tyas/
# ├── Zee/
# ├── Sam/
# └── 
# Urutan nama subfolder akan menjadi nama orang yang dikenali.
dataset_dir = './dataset' 

# --- Database untuk menyimpan embeddings wajah ---
database = {}

# --- Proses Dataset ---
for person_name in os.listdir(dataset_dir):  # Iterasi melalui setiap subfolder (misal: Muhammad Ilham)
    person_dir = os.path.join(dataset_dir, person_name)
    
    if os.path.isdir(person_dir):
        embeddings_list = []  # Tempat menyimpan embeddings wajah untuk orang ini

        # Mengambil gambar dari folder orang ini
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            
            if image_name.endswith('.jpeg') or image_name.endswith('.jpg'):
                # Baca gambar
                img = cv2.imread(image_path)

                # Deteksi wajah
                wajah = HaarCascade.detectMultiScale(img, 1.1, 4)
                
                if len(wajah) > 0:
                    x1, y1, width, height = wajah[0]
                    x1, y1 = abs(x1), abs(y1)
                    x2, y2 = x1 + width, y1 + height

                    # Ambil wajah dan ubah ukuran
                    gbr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    gbr = Image.fromarray(gbr)
                    gbr_array = asarray(gbr)
                    face = gbr_array[y1:y2, x1:x2]

                    face = Image.fromarray(face)
                    face = face.resize((160, 160))
                    face = asarray(face)

                    face = expand_dims(face, axis=0)
                    signature = MyFaceNet.embeddings(face)

                    # Tambahkan embeddings wajah ke daftar
                    embeddings_list.append(signature)

        # Simpan embeddings wajah orang ini ke dalam database
        if embeddings_list:
            database[person_name] = np.mean(embeddings_list, axis=0)  # Ambil rata-rata embeddings wajah

# Simpan database wajah ke dalam file data.pkl
with open('data2.pkl', 'wb') as file:
    pickle.dump(database, file)

print(f"Database wajah berhasil disimpan dalam 'data.pkl'. Total orang yang dikenali: {len(database)}")
