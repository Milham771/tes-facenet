from PIL import Image
import numpy as np
from numpy import asarray, expand_dims
import pickle
import cv2
from keras_facenet import FaceNet

# --- Inisialisasi Model ---
HaarCascade = cv2.CascadeClassifier(cv2.samples.findFile(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'))
MyFaceNet = FaceNet()

# --- Muat Database Wajah ---
with open("data.pkl", "rb") as myfile:
    database = pickle.load(myfile)

# --- Variabel untuk mengurangi lag ---
frame_counter = 0
process_every_n_frame = 5
identity = ' '
face_x1, face_y1, face_x2, face_y2 = 0, 0, 0, 0

cap = cv2.VideoCapture(0)

while(1):
    _, gbr1 = cap.read()
    if gbr1 is None:
        break

    frame_counter += 1

    # --- HANYA PROSES SETIAP N FRAME ---
    if frame_counter % process_every_n_frame == 0:
        wajah = HaarCascade.detectMultiScale(gbr1, 1.1, 4)

        if len(wajah) > 0:
            x1, y1, width, height = wajah[0]
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height

            gbr = cv2.cvtColor(gbr1, cv2.COLOR_BGR2RGB)
            gbr = Image.fromarray(gbr)
            gbr_array = asarray(gbr)
            face = gbr_array[y1:y2, x1:x2]

            face = Image.fromarray(face)
            face = face.resize((160, 160))
            face = asarray(face)

            face = expand_dims(face, axis=0)
            signature = MyFaceNet.embeddings(face)

            min_dist = 100
            found_identity = ' '
            for key, value in database.items():
                dist = np.linalg.norm(value - signature)
                if dist < min_dist:
                    min_dist = dist
                    found_identity = key
            
            # --- PENGECEKAN THRESHOLD ---
            # Jika jarak terdekat masih lebih besar dari 1.0, anggap tidak dikenal.
            # Anda bisa mengubah nilai 1.0 ini jika perlu.
            if min_dist < 1.0:
                identity = found_identity
            else:
                identity = "wajah tidak dikenal"
            
            # Simpan koordinat kotak
            face_x1, face_y1, face_x2, face_y2 = x1, y1, x2, y2

        else:
            identity = ' ' # Reset jika tidak ada wajah sama sekali

    # --- GAMBARKAN HASIL DI SETIAP FRAME ---
    if identity != ' ':
        # Tentukan warna teks: Merah jika tidak dikenal, Kuning jika dikenal
        text_color = (0, 0, 255) if identity == "wajah tidak dikenal" else (0, 255, 255)
        
        cv2.putText(gbr1, identity, (face_x1, face_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
        cv2.rectangle(gbr1, (face_x1, face_y1), (face_x2, face_y2), (0, 255, 0), 2)

    cv2.imshow('res', gbr1)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()