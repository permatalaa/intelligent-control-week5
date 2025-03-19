from ultralytics import YOLO
import cv2
import numpy as np

# Load model YOLOv8 Pose
model = YOLO("yolov8n-pose.pt")

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

# Indeks titik sendi berdasarkan skema COCO Keypoints
SELECTED_JOINTS = {
    0: "Hidung",
    1: "Mata Kiri", 2: "Mata Kanan",
    3: "Telinga Kiri", 4: "Telinga Kanan",
    5: "Bahu Kiri", 6: "Bahu Kanan",
    7: "Siku Kiri", 8: "Siku Kanan",
    9: "Pergelangan Tangan Kiri", 10: "Pergelangan Tangan Kanan",
    11: "Pinggul Kiri", 12: "Pinggul Kanan",
    13: "Lutut Kiri", 14: "Lutut Kanan",
    15: "Pergelangan Kaki Kiri", 16: "Pergelangan Kaki Kanan"
}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Deteksi pose
    results = model(frame)

    for result in results:
        keypoints = result.keypoints.xy.cpu().numpy()  # Ambil koordinat keypoints
        
        # Gambar titik-titik sendi yang dipilih dengan nama
        for i, name in SELECTED_JOINTS.items():
            if i < len(keypoints[0]):  # Pastikan indeks valid
                x, y = keypoints[0][i]
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)  # Warna hijau
                cv2.putText(frame, name, (int(x) + 5, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Tampilkan frame dengan titik-titik sendi tertentu
    cv2.imshow("YOLOv8 Pose Estimation - Titik Sendi Terpilih", frame)

    # Tambahkan pengecekan tombol 'q' untuk keluar dari loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bebaskan kamera dan tutup jendela OpenCV
cap.release()
cv2.destroyAllWindows()