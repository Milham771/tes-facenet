from PIL import Image
import numpy as np
from numpy import asarray, expand_dims
import pickle
import cv2
from keras_facenet import FaceNet
import time

# --- Inisialisasi Model ---
HaarCascade = cv2.CascadeClassifier(cv2.samples.findFile(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'))
MyFaceNet = FaceNet()

# --- Muat Database Wajah ---
with open("data2.pkl", "rb") as myfile:
    database = pickle.load(myfile)

# --- Variabel untuk mengurangi lag ---
frame_counter = 0
process_every_n_frame = 5  # Hanya proses pencocokan setiap N frame

# Cache untuk identitas wajah yang terdeteksi dengan struktur:
# {face_id: {'identity':..., 'min_dist':..., 'bbox':(x1,y1,x2,y2), 'timestamp':...} }
identity_cache = {}  

faces_info = []  # Menyimpan informasi wajah (bounding box dan identitas)

# Threshold perubahan posisi bounding box untuk memutuskan cache refresh
position_change_threshold = 20  # piksel

cache_valid_duration = 2.0  # detik, waktu maksimum cache dianggap valid

cap = cv2.VideoCapture(0)

def bbox_moved_enough(old_bbox, new_bbox, threshold):
    # Hitung jarak Euclidean antara koordinat tengah bounding box lama dan baru
    old_center = ((old_bbox[0] + old_bbox[2]) / 2, (old_bbox[1] + old_bbox[3]) / 2)
    new_center = ((new_bbox[0] + new_bbox[2]) / 2, (new_bbox[1] + new_bbox[3]) / 2)
    dist = np.linalg.norm(np.array(old_center) - np.array(new_center))
    return dist > threshold

while True:
    ret, gbr1 = cap.read()
    if not ret:
        break

    frame_counter += 1
    current_time = time.time()

    if frame_counter % process_every_n_frame == 0:
        gray = cv2.cvtColor(gbr1, cv2.COLOR_BGR2GRAY)# Konversi frame ke grayscale
        wajah = HaarCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(wajah) > 0:
            faces_info.clear()

            for (x1, y1, width, height) in wajah:
                x1, y1 = abs(x1), abs(y1)
                x2, y2 = x1 + width, y1 + height

                margin = 10
                x1 = max(x1 - margin, 0)
                y1 = max(y1 - margin, 0)
                x2 = x2 + margin
                y2 = y2 + margin

                face_id = f"{x1}-{y1}-{x2}-{y2}"
                new_bbox = (x1, y1, x2, y2)

                cache_entry = identity_cache.get(face_id)

                # Validasi cache: periksa apakah bbox berubah cukup banyak atau cache expired
                if (cache_entry is None or 
                    bbox_moved_enough(cache_entry['bbox'], new_bbox, position_change_threshold) or 
                    (current_time - cache_entry['timestamp']) > cache_valid_duration):
                    
                    # Lakukan ekstraksi dan pencocokan wajah ulang
                    face = gbr1[y1:y2, x1:x2]
                    face_img = Image.fromarray(face).resize((160, 160))
                    face_array = asarray(face_img)
                    face_array = expand_dims(face_array, axis=0)
                    signature = MyFaceNet.embeddings(face_array)

                    min_dist = 100
                    found_identity = ' '
                    for key, value in database.items():
                        dist = np.linalg.norm(value - signature)
                        if dist < min_dist:
                            min_dist = dist
                            found_identity = key

                    identity = found_identity if min_dist < 1.0 else "wajah tidak dikenal"

                    # Update cache
                    identity_cache[face_id] = {
                        'identity': identity,
                        'min_dist': min_dist,
                        'bbox': new_bbox,
                        'timestamp': current_time
                    }
                else:
                    identity = cache_entry['identity']
                    min_dist = cache_entry['min_dist']

                faces_info.append((x1, y1, x2, y2, identity, min_dist))
        else:
            faces_info.clear()

    # Gambarkan hasil di frame setiap waktu (tanpa harus proses ulang deteksi)
    for (x1, y1, x2, y2, identity, min_dist) in faces_info:
        text_color = (0, 0, 255) if identity == "wajah tidak dikenal" else (0, 255, 255)
        label = f"{identity} ({min_dist:.2f})"
        cv2.putText(gbr1, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
        cv2.rectangle(gbr1, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('Tekan ESC untuk menghentikan program', gbr1)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()

# =======================
# Alur Kerja Program:
# =======================
# 1. Program membuka kamera dan membaca frame secara berulang.
# 2. Setiap frame yang diambil dari kamera diubah menjadi format **grayscale**.
#    - Gambar **grayscale** digunakan karena deteksi wajah dengan Haar Cascade lebih efisien dan cepat dalam format ini.
# 3. Wajah yang terdeteksi oleh Haar Cascade disimpan dan diproses setiap 5 frame untuk mengurangi beban pemrosesan.
# 4. Setiap wajah yang terdeteksi akan diproses menggunakan model **FaceNet** untuk menghasilkan "signature" wajah.
#    - Program membandingkan signature wajah yang terdeteksi dengan database wajah menggunakan **jarak Euclidean** untuk pencocokan.
#    - Jika wajah dikenali dengan jarak yang cukup dekat (min_dist < 1.0), maka identitas wajah akan ditampilkan.
# 5. Hasil pengenalan wajah ditampilkan di gambar dengan kotak yang mengelilingi wajah yang terdeteksi.
#    - Kotak berwarna hijau menunjukkan wajah yang dikenali, dan kotak merah menunjukkan wajah yang tidak dikenali.
# 6. Program terus berjalan, menampilkan hasil deteksi wajah secara real-time.
#    - Program akan terus menampilkan wajah yang terdeteksi sampai tombol **ESC** ditekan untuk keluar dari program.
