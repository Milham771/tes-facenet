from PIL import Image
import numpy as np
from numpy import expand_dims
import pickle
import cv2
from keras_facenet import FaceNet
import tensorflow as tf

# Nonaktifkan log yang tidak perlu
tf.get_logger().setLevel('ERROR')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# --- Inisialisasi Model ---
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
MyFaceNet = FaceNet()

# --- Muat Database Wajah ---
with open("data3.pkl", "rb") as myfile:
    database = pickle.load(myfile)

# --- Variabel untuk Optimasi Kinerja ---
frame_counter = 0
process_every_n_frame = 5  # Proses setiap 5 frame untuk pengenalan untuk mengurangi beban
identity_cache = {}  # Cache untuk wajah yang dikenali
faces_info = []  # Daftar untuk menyimpan informasi wajah
last_processed_faces = []  # Cache wajah yang diproses untuk menghindari pemrosesan ulang
recognition_threshold = 0.6  # Threshold kepercayaan untuk pengenalan wajah

# --- Preprocessing Wajah untuk FaceNet ---
def process_face(face_img):
    """Proses wajah untuk pengenalan menggunakan FaceNet."""
    face_img = Image.fromarray(face_img).resize((160, 160))
    face_img = np.array(face_img)
    face_img = expand_dims(face_img, axis=0)  # Menambahkan dimensi untuk batch size
    signature = MyFaceNet.embeddings(face_img)
    return signature

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Menyesuaikan resolusi untuk pemrosesan yang lebih lancar
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Konversi frame ke grayscale
    # coba bandingkan bagusan mana pakai grayscale atau RGB, dari yang ku baca di internet, haar cascade lebih baik pakai grayscale
    # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Konversi frame ke RGB untuk FaceNet
    # --- Deteksi Wajah (mengurangi frekuensi deteksi untuk pelacakan yang lebih lancar) ---
    faces = face_detector.detectMultiScale(gray_frame, 1.1, 4)  # Deteksi wajah menggunakan Haar Cascade

    current_faces = []
    for (x, y, w, h) in faces:
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = x1 + w, y1 + h

        # Menambahkan margin (persentase dari ukuran wajah)
        margin = 0
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = x2 + margin
        y2 = y2 + margin

        current_faces.append((x1, y1, x2, y2))

    # --- Pengenalan Wajah (hanya proses wajah baru atau yang bergerak) ---
    if frame_counter % process_every_n_frame == 0 or not last_processed_faces:
        faces_info = []
        for (x1, y1, x2, y2) in current_faces:
            face_id = f"{x1}-{y1}-{x2}-{y2}"

            # Memeriksa apakah wajah bergerak signifikan dari posisi sebelumnya
            moved_significantly = True
            for (lx1, ly1, lx2, ly2) in last_processed_faces:
                if abs(x1 - lx1) < 20 and abs(y1 - ly1) < 20:
                    moved_significantly = False
                    break

            if face_id not in identity_cache or moved_significantly:
                try:
                    face_img = frame[y1:y2, x1:x2]
                    if face_img.size == 0:
                        continue
                    
                    signature = process_face(face_img)
                    
                    # Pencocokan wajah menggunakan Cosine Similarity
                    min_dist = float('inf')
                    found_identity = 'Unknown'

                    for key, db_sig in database.items():
                        # Hitung cosine similarity
                        cos_similarity = np.dot(signature, db_sig.T) / (np.linalg.norm(signature) * np.linalg.norm(db_sig))
                        dist = 1 - cos_similarity  # Konversi cosine similarity ke dalam jarak

                        if dist < min_dist:
                            min_dist = dist
                            found_identity = key
                    
                    # Terapkan threshold untuk pengenalan
                    identity = found_identity if min_dist < recognition_threshold else "Unknown"
                    identity_cache[face_id] = (identity, min_dist)
                except Exception as e:
                    print(f"Error dalam pemrosesan wajah: {e}")
                    identity = "Error"
                    identity_cache[face_id] = (identity, float('inf'))

            faces_info.append((x1, y1, x2, y2, identity_cache[face_id][0], identity_cache[face_id][1]))

        last_processed_faces = current_faces.copy()

    # --- Gambar Hasil ---
    for (x1, y1, x2, y2, identity, dist) in faces_info:
        dist = float(dist)

        # Pewarnaan dinamis berdasarkan tingkat kepercayaan
        if identity == "Unknown" or identity == "Error":
            color = (0, 0, 255)  # Merah
            thickness = 2
        else:
            # Hijau dengan intensitas berdasarkan tingkat kepercayaan
            confidence = max(0, min(1, 1 - (dist / recognition_threshold)))
            green = int(255 * confidence)
            color = (0, green, 0)
            thickness = 2 + int(2 * confidence)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Tampilkan identitas dan jarak
        label = f"{identity} ({dist:.2f})" if identity not in ["Unknown", "Error"] else identity
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow('Face Recognition (Press ESC to quit)', frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
# =======================
# Alur Kerja Program:
# =======================
# 1. Program pertama-tama membuka kamera dan mulai membaca frame secara terus-menerus.
# 2. Setiap frame yang diambil dari kamera akan diubah ke format RGB atau Grayscale (Pilih dengan sesuai yang diaktifkan diatas).
#    - Jika menggunakan format RGB, gambar memiliki lebih banyak informasi warna untuk pengenalan wajah dengan FaceNet.
#    - Jika menggunakan Grayscale, gambar hanya terdiri dari satu saluran warna (hitam-putih), yang bisa lebih cepat tetapi dengan akurasi yang sedikit berkurang.
# 3. Setelah gambar diproses, program akan mendeteksi wajah menggunakan algoritma Haar Cascade.
#    - Wajah yang terdeteksi akan disimpan dan diproses setiap 5 frame untuk menghindari pemrosesan berulang dan mengurangi beban komputasi.
# 4. Proses pengenalan wajah dilakukan dengan membandingkan "signature" wajah yang terdeteksi dengan database wajah yang sudah ada.
#    - Pembandingan dilakukan menggunakan cosine similarity, yang mengukur seberapa mirip wajah yang terdeteksi dengan wajah yang ada di database.
# 5. Hasil pengenalan wajah akan ditampilkan pada gambar dengan kotak berwarna yang mengelilingi wajah yang terdeteksi.
#    - Kotak ini akan berwarna hijau jika wajah dikenali dengan tingkat kepercayaan tinggi dan merah jika wajah tidak dikenali atau terdapat kesalahan.
# 6. Program akan terus berjalan, menampilkan hasil pengenalan wajah secara real-time di jendela OpenCV.
#    - Program akan terus berjalan hingga pengguna menekan tombol ESC untuk keluar dari program.

