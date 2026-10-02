import ultralytics
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Model yolu
model_path = "pelvis_asıl.pt"
model = YOLO(model_path)

# Test görüntüsünün yolu
test_image_path = "Test_Resimleri/462.jpg"

# Sonuçları kaydetmek için
save_dir = "results/"
os.makedirs(save_dir, exist_ok=True)

results = model(test_image_path, conf=0.3)


def calculate_outer_angle(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cos_theta = dot_product / (norm_v1 * norm_v2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle = np.arccos(cos_theta)
    angle_deg = np.degrees(angle)
    if angle_deg < 180:
        angle_deg = 180 - angle_deg
    return angle_deg

for result in results:
    img_with_circles = cv2.imread(test_image_path)

    pelvic_point = None
    suorcil_point = None
    for box in result.boxes:
        cls = int(box.cls[0])  # Sınıf ID'si
        label = result.names[cls]

        print(f"Detected label: {label}")

        if label == "FEMUR":
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if x1 > img_with_circles.shape[1] // 2:
                continue

            femur_head_center_x = x2
            femur_head_center_y = y1 + (y2 - y1) // 4

            radius_x = (x2 - x1) // 4
            radius_y = (y2 - y1) // 4
            radius = min(radius_x, radius_y)
            radius = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)

            femur_head_center_x = min(femur_head_center_x, x2 - radius)
            femur_head_center_y = min(femur_head_center_y, y1 + radius)


            cv2.circle(img_with_circles, (femur_head_center_x, femur_head_center_y), radius, (255, 0, 0), 3)
            cv2.circle(img_with_circles, (femur_head_center_x, femur_head_center_y), 5, (0, 255, 0), -1)


            offset = 20
            pelvic_point = (int((x1 + x2) / 2) + offset, y1)

            top_point = (femur_head_center_x, femur_head_center_y - radius)
            cv2.line(img_with_circles, (femur_head_center_x, femur_head_center_y), top_point, (0, 255, 0), 2)

            femur_vector = np.array([0, femur_head_center_y - y1])
            pelvic_vector = np.array([pelvic_point[0] - femur_head_center_x, pelvic_point[1] - femur_head_center_y])

            lcea_angle = calculate_outer_angle(femur_vector, pelvic_vector)

        if label == "SUORCIL" and suorcil_point is None:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if x1 > img_with_circles.shape[1] // 2:
                continue

            print(f"SUORCIL box: {x1}, {y1}, {x2}, {y2}")

            offset = 20
            suorcil_point = (x1, y1 + offset)

            cv2.circle(img_with_circles, suorcil_point, 5, (0, 0, 255), -1)

            cv2.rectangle(img_with_circles, (x1, y1), (x2, y2), (0, 0, 255), 2)

            suorcil_vector = np.array([suorcil_point[0] - femur_head_center_x, suorcil_point[1] - femur_head_center_y])

            suorcil_angle = calculate_outer_angle(femur_vector, suorcil_vector)
            cv2.putText(img_with_circles, f"LCEA: {suorcil_angle:.2f}°",
                        (suorcil_point[0] + 10, suorcil_point[1] + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.line(img_with_circles, (femur_head_center_x, femur_head_center_y), suorcil_point, (0, 0, 255), 2)


    output_path = os.path.join(save_dir, f"result_{os.path.basename(test_image_path)}")
    cv2.imwrite(output_path, img_with_circles)

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img_with_circles, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

print("Model test işlemi tamamlandı. Sonuçlar kaydedildi: ", save_dir)
