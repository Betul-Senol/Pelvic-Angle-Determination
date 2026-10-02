import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Model yükleme
model_path = "pelvis_asıl.pt"
model = YOLO(model_path)

# Test görüntüsünün yolu
test_image_path = "Test_Resimleri/478.jpg"

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

                # Sağ femur için daireyi çiz
                cv2.circle(img_with_circles, (femur_head_center_x, femur_head_center_y), radius, (255, 0, 0), 3)
                cv2.circle(img_with_circles, (femur_head_center_x, femur_head_center_y), 5, (0, 255, 0), -1)

        elif label == "SUORCIL":
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

                    teardrop_bottom_x = x1 + (x2 - x1) // 2
                    teardrop_bottom_y = y2
                    cv2.circle(img_with_circles, (teardrop_bottom_x, teardrop_bottom_y), 5, (0, 255, 255), -1)
                    teardrop_point = (teardrop_bottom_x, teardrop_bottom_y)


    if suorcil_point and teardrop_point:
        cv2.line(img_with_circles, suorcil_point, teardrop_point, (0, 255, 255), 2)
        vector1 = np.array([suorcil_point[0] - teardrop_point[0], suorcil_point[1] - teardrop_point[1]])
        vector2 = np.array([1, 0])


        dot_product = np.dot(vector1, vector2)

        magnitude_vector1 = np.linalg.norm(vector1)
        magnitude_vector2 = np.linalg.norm(vector2)

        cosine_theta = dot_product / (magnitude_vector1 * magnitude_vector2)
        angle_radians = np.arccos(np.clip(cosine_theta, -1.0, 1.0))
        angle_degrees = np.degrees(angle_radians)

        print(f"Sharp Angle: {angle_degrees:.2f} derece")


        if suorcil_point:
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f"Sharp Açı: {angle_degrees:.2f}°"
            cv2.putText(img_with_circles, text, (int(suorcil_point[0] + 20), int(suorcil_point[1] - 10)), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    if teardrop_point:
        teardrop_x, teardrop_y = teardrop_point
        line_length = 700

        line_start_x = max(0, teardrop_x - line_length // 2)

        line_end_x = min(img_with_circles.shape[1], teardrop_x + line_length // 2)
        cv2.line(img_with_circles, (line_start_x, teardrop_y), (line_end_x, teardrop_y), (255, 0, 0), 2)


    output_path = os.path.join(save_dir, f"result_{os.path.basename(test_image_path)}")
    cv2.imwrite(output_path, img_with_circles)


    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img_with_circles, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

print("Model test işlemi tamamlandı. Sonuçlar kaydedildi: ", save_dir)
