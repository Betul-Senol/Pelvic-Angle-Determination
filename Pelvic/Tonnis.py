import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from ultralytics import YOLO
import math

# Model yükleme
model_path = "pelvis_asıl.pt"
model = YOLO(model_path)

# Test görüntüsünün yolu
test_image_path = "Test_Resimleri/466.jpg"

# Sonuçların kaydedileceği dizin
save_dir = "path_to_save_results"
os.makedirs(save_dir, exist_ok=True)

# Görüntüyü yükle
image = cv2.imread(test_image_path)
results = model.predict(source=test_image_path, save=False, conf=0.2)


for result in results:
    img_with_circles = cv2.imread(test_image_path)

    suorcil_boxes = []
    teardrop_boxes = []
    suorcil_point = None
    teardrop_point = None

    for box in result.boxes:
        cls = int(box.cls[0])
        label = result.names[cls]

        print(f"Detected label: {label}")

        if label == "FEMUR":
            x1, y1, x2, y2 = map(int, box.xyxy[0])


            if x1 > img_with_circles.shape[1] // 2:

                femur_head_center_x = x1
                femur_head_center_y = y1 + (y2 - y1) // 4

                radius_x = (x2 - x1) // 4
                radius_y = (y2 - y1) // 4
                radius = min(radius_x, radius_y)
                radius = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)

                femur_head_center_x = max(femur_head_center_x, x1 + radius)
                femur_head_center_y = max(femur_head_center_y, y1 + radius)

                femur_head_center_x = x1 + radius
                femur_head_center_y = y1 + radius

                cv2.circle(img_with_circles, (femur_head_center_x, femur_head_center_y), radius, (255, 0, 0), 3)
                cv2.circle(img_with_circles, (femur_head_center_x, femur_head_center_y), 5, (0, 255, 0), -1)

        elif label == "SUORCIL":  # SUORCIL sınıfı
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if x1 > img_with_circles.shape[1] // 2:
                overlap = False
                for (prev_x1, prev_y1, prev_x2, prev_y2) in suorcil_boxes:
                    if not (x2 < prev_x1 or x1 > prev_x2 or y2 < prev_y1 or y1 > prev_y2):
                        overlap = True
                        break

                if not overlap:

                    suorcil_rightmost_x = x2
                    suorcil_rightmost_y = y1 + (y2 - y1) // 2
                    cv2.circle(img_with_circles, (suorcil_rightmost_x, suorcil_rightmost_y), 5, (255, 255, 0), -1)
                    suorcil_point = (suorcil_rightmost_x, suorcil_rightmost_y)

        elif label == "TEARDROP":
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if x1 > img_with_circles.shape[1] // 2:
                overlap = False
                for (prev_x1, prev_y1, prev_x2, prev_y2) in teardrop_boxes:
                    if not (x2 < prev_x1 or x1 > prev_x2 or y2 < prev_y1 or y1 > prev_y2):
                        overlap = True
                        break

                if not overlap:

                    teardrop_top_x = x1 + (x2 - x1) // 2
                    teardrop_top_y = y1
                    cv2.circle(img_with_circles, (teardrop_top_x, teardrop_top_y), 5, (0, 255, 255), -1)

                    teardrop_point = (teardrop_top_x, teardrop_top_y)

    if suorcil_point and teardrop_point:
        cv2.line(img_with_circles, suorcil_point, teardrop_point, (255, 255, 0), 2)

    if teardrop_point:
        teardrop_top_y = teardrop_point[1]
        img_width = img_with_circles.shape[1]

        left_limit = max(0, teardrop_point[0] - 0)
        right_limit = min(img_width, teardrop_point[0] + 500)


        cv2.line(img_with_circles, (left_limit, teardrop_top_y), (right_limit, teardrop_top_y), (255, 0, 0), 2)


    if suorcil_point and teardrop_point:

        horizontal_vector = (right_limit - left_limit, 0)
        teardrop_suorcil_vector = (suorcil_point[0] - teardrop_point[0], suorcil_point[1] - teardrop_point[1])
        dot_product = (horizontal_vector[0] * teardrop_suorcil_vector[0]) + (horizontal_vector[1] * teardrop_suorcil_vector[1])
        horizontal_vector_magnitude = math.sqrt(horizontal_vector[0]**2 + horizontal_vector[1]**2)
        teardrop_suorcil_vector_magnitude = math.sqrt(teardrop_suorcil_vector[0]**2 + teardrop_suorcil_vector[1]**2)

        cos_theta = dot_product / (horizontal_vector_magnitude * teardrop_suorcil_vector_magnitude)
        angle_rad = math.acos(cos_theta)
        angle_deg = math.degrees(angle_rad)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_with_circles, f"Tonnis Angle: {angle_deg:.2f} degrees",
                    (femur_head_center_x - 50, femur_head_center_y - 50),
                    font, 1, (0, 255, 0), 2)

    # Sonuçları kaydet
    output_path = os.path.join(save_dir, f"result_{os.path.basename(test_image_path)}")
    cv2.imwrite(output_path, img_with_circles)

    # Görüntüyü göster
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img_with_circles, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

print("Model test işlemi tamamlandı. Sonuçlar kaydedildi: ", save_dir)
